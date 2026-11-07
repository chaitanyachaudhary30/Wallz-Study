import {
  ActionJson,
  AgentInfo,
  ConfigDefaults,
  Explanation,
  GameConfig,
  ScenarioDetail,
  SessionData,
} from './types';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (res.status === 204) {
    return {} as T;
  }

  const json = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(res.status, json.error || `HTTP error ${res.status}`, json);
  }
  return json as T;
}

export const api = {
  // Config & Agents
  fetchConfigDefaults: () => request<ConfigDefaults>('/api/config/defaults'),
  fetchAgents: (cfg: Partial<GameConfig>) => {
    const params = new URLSearchParams();
    if (cfg.board_size) params.set('N', String(cfg.board_size));
    if (cfg.A !== undefined) params.set('A', String(cfg.A));
    if (cfg.B !== undefined) params.set('B', String(cfg.B));
    if (cfg.C !== undefined) params.set('C', String(cfg.C));
    if (cfg.K !== undefined) params.set('K', String(cfg.K));
    if (cfg.M !== undefined) params.set('M', String(cfg.M));
    return request<AgentInfo[]>(`/api/agents?${params.toString()}`);
  },

  // Game Management
  createGame: (config: GameConfig, p0: string, p1: string) =>
    request<SessionData>('/api/games', {
      method: 'POST',
      body: JSON.stringify({ config, p0, p1 }),
    }),
  fetchGame: (id: string) => request<SessionData>(`/api/games/${id}`),
  sendAction: (id: string, action: ActionJson) =>
    request<SessionData>(`/api/games/${id}/action`, {
      method: 'POST',
      body: JSON.stringify({ action }),
    }),
  stepGame: (id: string, explain: boolean = true) =>
    request<{ action: ActionJson; explanation: Explanation | null; session: SessionData }>(
      `/api/games/${id}/step`,
      {
        method: 'POST',
        body: JSON.stringify({ explain }),
      }
    ),
  undoGame: (id: string, n: number = 1, toHuman: boolean = false) =>
    request<SessionData>(`/api/games/${id}/undo`, {
      method: 'POST',
      body: JSON.stringify({ n, to_human: toHuman }),
    }),
  deleteGame: (id: string) =>
    request<void>(`/api/games/${id}`, { method: 'DELETE' }),

  // Explain / Analyze
  analyzePosition: (id: string, agentSpec: string) =>
    request<Explanation>(`/api/games/${id}/analyze`, {
      method: 'POST',
      body: JSON.stringify({ agent: agentSpec }),
    }),

  // Arena
  runArenaBatch: (config: GameConfig, p0: string, p1: string, n: number, seed: number) =>
    request<any>('/api/arena/batch', {
      method: 'POST',
      body: JSON.stringify({ config, p0, p1, n, seed }),
    }),
  fetchArenaBatch: (bid: string) => request<any>(`/api/arena/batch/${bid}`),
  fetchArenaGame: (bid: string, idx: number) => request<any>(`/api/arena/batch/${bid}/${idx}`),

  // Replays
  fetchReplays: (offset: number = 0, limit: number = 50, q?: string) => {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (q) params.set('q', q);
    return request<{ total: number; items: any[] }>(`/api/replays?${params.toString()}`);
  },
  fetchReplayItem: (file: string, index: number) =>
    request<any>(`/api/replays/item?file=${encodeURIComponent(file)}&index=${index}`),
  uploadReplay: (data: any) =>
    request<any>('/api/replays/upload', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Scenarios
  fetchScenarios: () => request<any[]>('/api/scenarios'),
  fetchScenario: (name: string) => request<ScenarioDetail>(`/api/scenarios/${name}`),
  createGameFromScenario: (name: string, p0: string = 'human', p1: string = 'human') =>
    request<SessionData>(`/api/games/from-scenario/${name}`, {
      method: 'POST',
      body: JSON.stringify({ p0, p1 }),
    }),

  // Training Data
  fetchTrainingRuns: () => request<any[]>('/api/training/runs'),
  fetchTrainingRun: (id: string) => request<any>(`/api/training/runs/${id}`),
  fetchTrainingSummary: () => request<any>('/api/training/summary'),
  fetchTrainingConfig: () => request<any>('/api/training/config'),

  // Rules Documentation
  fetchRules: () => request<{ rules: string }>('/api/rules'),

  // Live Training (P15)
  startLiveTraining: (steps: number = 2000, seed: number = 42) =>
    request<any>('/api/training/live', {
      method: 'POST',
      body: JSON.stringify({ steps, seed }),
    }),
  fetchLiveStatus: () => request<any>('/api/training/live/status'),
  stopLiveTraining: () => request<any>('/api/training/live/stop', { method: 'POST' }),
};
