// A Sokoban solver for make.fun's sokoban-solver-01: one program answers 5 levels, each in as few pushes as it can.
//
// The idea, in one paragraph: every level the generator draws is a 1-wide hallway with 1-4 rooms off it, and a box
// pushed out of a room into the hallway can never come back, so each room is an independent problem and the level's
// push-optimum is the sum of its rooms' optima. I cut the hallway's dead squares out of the floor, solve each room
// on its own with an A* over pushes, and stitch the rooms' move strings together with the player's walks between
// them. The A* state is the boxes' squares plus the player's reachable region; its estimate is the cheapest
// assignment of boxes to goals by push-distance (a lower bound, so the first solved state popped is push-optimal),
// and it prunes dead squares (a box that can never reach a goal) and frozen 2x2 blocks. No clock is ever read and
// nothing is random: the budget is counted in expanded states, so two runs write the same bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The four moves: [dx, dy, walk letter, push letter]. A fixed order everywhere, so every tie breaks the same way.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const WALK = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

/** A level: width, height, walls, goals, boxes (sorted) and the player, all as square indexes y * w + x. */
type Level = { w: number; h: number; wall: Uint8Array; goals: number[]; boxes: number[]; player: number };

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player, + player on goal). */
function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The fewest pushes a box on each square needs to reach each goal, ignoring the other boxes and the player: a
 * reverse breadth-first search from every goal, where a box at x could have been pushed from x - d when the player
 * could stand at x - 2d. dist[g * size + s] is that count for goal g, -1 when no push path exists. A square with no
 * goal in reach is a dead square: a box there can never be saved.
 */
function pushDistances(L: Level): Int32Array {
  const size = L.w * L.h;
  const dist = new Int32Array(L.goals.length * size).fill(-1);
  const queue = new Int32Array(size);
  for (let gi = 0; gi < L.goals.length; gi++) {
    const base = gi * size;
    const g = L.goals[gi]!;
    let head = 0;
    let tail = 0;
    queue[tail++] = g;
    dist[base + g] = 0;
    while (head < tail) {
      const x = queue[head++]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * L.w + DX[k]!;
        const y = x - d;
        const behind = x - 2 * d;
        if (L.wall[y] || L.wall[behind] || dist[base + y] >= 0) continue;
        dist[base + y] = dist[base + x]! + 1;
        queue[tail++] = y;
      }
    }
  }
  return dist;
}

/** The cheapest way to match every box to a different goal by push-distance (Infinity when no matching exists). */
function matching(boxes: number[], dist: Int32Array, nGoals: number, size: number): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 5) {
    // Many boxes: the sum of each box's nearest goal, a weaker but cheap lower bound.
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (let g = 0; g < nGoals; g++) {
        const d = dist[g * size + b]!;
        if (d >= 0 && d < m) m = d;
      }
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  }
  // Few boxes: the exact cheapest assignment, by trying the goals box by box with branch and bound.
  const used = new Uint8Array(nGoals);
  let best = Infinity;
  const assign = (i: number, cost: number): void => {
    if (cost >= best) return;
    if (i === n) {
      best = cost;
      return;
    }
    for (let g = 0; g < nGoals; g++) {
      if (used[g]) continue;
      const d = dist[g * size + boxes[i]!]!;
      if (d < 0) continue;
      used[g] = 1;
      assign(i + 1, cost + d);
      used[g] = 0;
    }
  };
  assign(0, 0);
  return best;
}

/**
 * The squares the player reaches from `from` with `occupied` in the way, marked in `seen` with `stamp`; the lowest.
 * The lowest square names the region: two player squares in one region are the same state.
 */
function regionLow(L: Level, occupied: Uint8Array, from: number, seen: Int32Array, stamp: number, stack: Int32Array): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * L.w + DX[k]!;
      if (L.wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(L: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const w = L.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(L.wall[a] || occupied[a]) || !(L.wall[b] || occupied[b]) || !(L.wall[c] || occupied[c]) || !(L.wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** A state's key: its region's lowest square and its boxes' squares (a number while it fits, else a string). */
function keyOf(boxes: number[], low: number, size: number): number | string {
  if (Math.pow(size, boxes.length + 1) < Number.MAX_SAFE_INTEGER) {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  }
  return low + ':' + boxes.join(',');
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters, or null when unreachable. */
function walkTo(L: Level, boxes: number[], from: number, to: number): string | null {
  if (from === to) return '';
  const size = L.w * L.h;
  const occupied = new Uint8Array(size);
  for (const b of boxes) occupied[b] = 1;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * L.w + DX[k]!;
      if (L.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = WALK[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, the case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) p += ch === 'u' || ch === 'U' ? -w : ch === 'd' || ch === 'D' ? w : ch === 'l' || ch === 'L' ? -1 : 1;
  return p;
}

const MAX_EXPANSIONS = 500_000;

/**
 * The push-optimal solution of a level: {pushes, moves} (moves in LURD, an upper letter a push), or null when the
 * search gives up. A* over pushes: a state is the boxes plus the player's region, the estimate is the matching
 * lower bound (admissible and consistent, so the first solved state popped is optimal and no state is reopened),
 * the heap breaks ties toward more pushes done, and the budget is counted in expansions, never the clock.
 */
function searchPushOptimal(L: Level, maxExpansions: number = MAX_EXPANSIONS): { pushes: number; moves: string } | null {
  const size = L.w * L.h;
  const goal = new Uint8Array(size);
  for (const g of L.goals) goal[g] = 1;
  const dist = pushDistances(L);
  const live = (s: number): boolean => {
    for (let g = 0; g < L.goals.length; g++) if (dist[g * size + s]! >= 0) return true;
    return false;
  };
  if (L.boxes.some((b) => !live(b))) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);

  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  // The states in the order found: boxes, the player's square, parent, and the push that made them.
  const sBoxes: number[][] = [L.boxes];
  const sPlayer: number[] = [L.player];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1];
  const sTo: number[] = [-1];
  const sG: number[] = [0];
  const sKey: (number | string)[] = [];
  const best = new Map<number | string, number>();
  for (const b of L.boxes) occupied[b] = 1;
  const firstLow = regionLow(L, occupied, L.player, seen, ++stamp, stack);
  for (const b of L.boxes) occupied[b] = 0;
  const firstKey = keyOf(L.boxes, firstLow, size);
  sKey.push(firstKey);
  best.set(firstKey, 0);

  // The open list: a binary heap on [f, g, state], packed flat, ties toward more pushes done, then earlier found.
  const heap: number[] = [];
  const heapLess = (a: number, b: number): boolean => {
    const fa = heap[a]!;
    const fb = heap[b]!;
    if (fa !== fb) return fa < fb;
    const ga = heap[a + 1]!;
    const gb = heap[b + 1]!;
    if (ga !== gb) return ga > gb;
    return heap[a + 2]! < heap[b + 2]!;
  };
  const heapPush = (f: number, g: number, s: number): void => {
    let i = heap.length;
    heap.push(f, g, s);
    while (i > 0) {
      const p = ((i / 3 - 1) >> 1) * 3;
      if (!heapLess(i, p)) break;
      for (let k = 0; k < 3; k++) {
        const t = heap[i + k]!;
        heap[i + k] = heap[p + k]!;
        heap[p + k] = t;
      }
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[2]!;
    const last = heap.length - 3;
    if (last > 0) {
      heap[0] = heap[last]!;
      heap[1] = heap[last + 1]!;
      heap[2] = heap[last + 2]!;
    }
    heap.length = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 3;
      if (c >= heap.length) break;
      if (c + 3 < heap.length && heapLess(c + 3, c)) c += 3;
      if (!heapLess(c, i)) break;
      for (let k = 0; k < 3; k++) {
        const t = heap[i + k]!;
        heap[i + k] = heap[c + k]!;
        heap[c + k] = t;
      }
      i = c;
    }
    return top;
  };
  const h0 = matching(L.boxes, dist, L.goals.length, size);
  if (h0 === Infinity) return null;
  heapPush(h0, 0, 0);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const s = heapPop();
    const g = sG[s]!;
    if (best.get(sKey[s]!) !== g) continue; // a better way to this state was found since: stale entry
    const boxes = sBoxes[s]!;
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (expanded >= maxExpansions) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    regionLow(L, occupied, sPlayer[s]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * L.w + DX[k]!;
        const t = b + d;
        if (seen[b - d] !== here || L.wall[t] || occupied[t] || !live(t)) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(L, occupied, goal, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((a, b2) => a - b2);
          const low = regionLow(L, occupied, b, mark, ++stamp, stack);
          key = keyOf(next, low, size);
          h = matching(next, dist, L.goals.length, size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity) continue;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= g + 1) continue;
        best.set(key, g + 1);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(g + 1);
        sKey.push(key);
        heapPush(g + 1 + h, g + 1, sBoxes.length - 1);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;
  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = L.player;
  let boxes = L.boxes;
  for (const s of path) {
    const from = sFrom[s]!;
    const to = sTo[s]!;
    const d = to - from;
    const k = DX.findIndex((dx, i) => DY[i]! * L.w + dx === d);
    const walk = walkTo(L, boxes, player, from - d);
    if (walk === null || k < 0) return null;
    moves += walk + PUSH[k]!;
    player = from;
    boxes = sBoxes[s]!;
  }
  return { pushes: path.length, moves };
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal) out of
 * the floor, and each part left that holds a box is a room. Null when a box starts in the hallway, a goal lies
 * outside every part, or some part holds more or fewer boxes than goals.
 */
function roomsOf(L: Level): { squares: number[]; boxes: number[]; goals: number[]; done: boolean }[] | null {
  const size = L.w * L.h;
  const dist = pushDistances(L);
  const live = (s: number): boolean => {
    for (let g = 0; g < L.goals.length; g++) if (dist[g * size + s]! >= 0) return true;
    return false;
  };
  const row = Math.floor(L.player / L.w);
  const cut = new Uint8Array(size);
  for (let x = 0; x < L.w; x++) {
    const i = row * L.w + x;
    if (!L.wall[i] && !live(i)) cut[i] = 1;
  }
  if (L.boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: { squares: number[]; boxes: number[]; goals: number[]; done: boolean }[] = [];
  for (let s = 0; s < size; s++) {
    if (L.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i]! + DY[k]! * L.w + DX[k]!;
        if (!L.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of L.boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of L.goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn, the other rooms' boxes standing as
 * walls (on their goals once their room is done), the player walking on from where the last room left it. A box
 * never leaves its room, so the rooms' optima add up to the level's. Falls back to one search over the whole level
 * when the level is no set of rooms.
 */
function solve(text: string): string {
  const L = parse(text);
  const rooms = roomsOf(L);
  if (rooms !== null) {
    let player = L.player;
    let moves = '';
    for (const room of rooms) {
      const wall = L.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub: Level = {
        w: L.w,
        h: L.h,
        wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      const solved = searchPushOptimal(sub);
      if (solved === null) return '';
      moves += solved.moves;
      player = endOf(L.w, player, solved.moves);
      room.done = true;
    }
    return moves;
  }
  const solved = searchPushOptimal(L);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
