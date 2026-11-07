export type ActionType = 'MOVE' | 'HOP' | 'WALL' | 'REPLACE' | 'BLOCK';

export type ActionJson = [ActionType, any];

export interface WallState {
  r: number;
  c: number;
  o: 'H' | 'V';
  owner: number;
  latest: boolean;
}

export interface BlockedCellState {
  r: number;
  c: number;
  remaining: number;
}

export interface TerminalState {
  over: boolean;
  winner: number | null;
  reason: 'goal' | 'no_legal_actions' | 'repetition' | 'truncated' | 'timeout' | null;
}

export interface GameRules {
  A: boolean;
  B: boolean;
  C: boolean;
  K: number;
  M: number;
}

export interface GameConfig {
  board_size: number;
  max_plies?: number;
  A: boolean;
  B: boolean;
  C: boolean;
  K: number;
  M: number;
}

export interface LegalActionsMap {
  MOVE: [number, number][];
  HOP: [number, number][];
  WALL: [number, number, 'H' | 'V'][];
  REPLACE: [number, number, 'H' | 'V'][];
  BLOCK: [number, number][];
}

export interface GameState {
  n: number;
  rules: GameRules;
  pawns: [[number, number], [number, number]];
  goals: [number, number];
  walls: WallState[];
  inventory: [number, number];
  hop: [boolean, boolean];
  block: [boolean, boolean];
  blocked: BlockedCellState[];
  to_move: number;
  turn: number;
  terminal: TerminalState;
  paths: [number, number][][];
  legal: LegalActionsMap;
  legal_count: number;
}

export interface HistoryEntry {
  ply: number;
  player: number;
  action: ActionJson;
  source: string;
  explanation: Explanation | null;
  decision_seconds: number;
}

export interface SessionData {
  id: string;
  config: GameConfig;
  specs: { [key: string]: string };
  state: GameState;
  history: HistoryEntry[];
  created: number;
}

export interface OverlayItem {
  action: ActionJson;
  value: number | null;
  label: string;
}

export interface Explanation {
  agent: string;
  spec: string;
  player: number;
  chosen: ActionJson | null;
  decision_seconds: number;
  summary: string;
  details: Record<string, any>;
  overlay: OverlayItem[];
}

export interface AgentInfo {
  spec: string;
  name: string;
  family: 'random' | 'greedy' | 'minimax' | 'dqn' | 'exact' | 'human';
  label: string;
  available: boolean;
  reason: string | null;
  explainable: boolean;
}

export interface ConfigDefaults {
  defaults: Record<string, any>;
  limits: {
    N: [number, number];
    K: [number, number];
    M: [number, number];
  };
  presets: Array<{
    id: string;
    label: string;
    config: GameConfig;
  }>;
}

export interface ScenarioCheck {
  key: string;
  ok: boolean;
  message: string;
}

export interface ScenarioFrame {
  ply: number;
  action: ActionJson | null;
  state: GameState;
  checks: ScenarioCheck[];
}

export interface ScenarioDetail {
  scenario: {
    name: string;
    title: string;
    description: string;
    config: GameConfig;
    script: ActionJson[];
    expect: any[];
    final: any;
  };
  frames: ScenarioFrame[];
}
