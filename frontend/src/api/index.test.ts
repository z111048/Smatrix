import { analyzeStructure, healthCheck } from './index';

const mockFetch = (response: Partial<Response>) => {
  const fetchMock = vi.fn().mockResolvedValue(response);
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('analyzeStructure', () => {
  it('posts the backend payload shape with SI-valued fields and defaults', async () => {
    const analysisResult = {
      success: true,
      displacements: [],
      reactions: [],
      internal_forces: []
    };
    const fetchMock = mockFetch({
      ok: true,
      json: vi.fn().mockResolvedValue(analysisResult)
    });

    const result = await analyzeStructure(
      [
        { id: 1, x: 0, y: 0, support: 'fixed' },
        { id: 2, x: 3, y: 0, support: 'roller' }
      ],
      [
        { id: 1, nodeI: 1, nodeJ: 2, E: 210e9, I: 8.5e-6, releaseI: false, releaseJ: true }
      ],
      [
        { nodeId: 2, Fy: -12000, Mz: 2500 }
      ],
      [
        { id: 4, elementId: 1, w1: -1500, w2: -2500 }
      ],
      [
        { id: 5, elementId: 1, a: 1.25, Fx: 3000, Fy: -4000 }
      ]
    );

    expect(result).toBe(analysisResult);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://localhost:8000/analyze');
    expect(options.method).toBe('POST');
    expect(options.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(String(options.body))).toEqual({
      nodes: [
        { id: 1, x: 0, y: 0, support: 'fixed' },
        { id: 2, x: 3, y: 0, support: 'roller' }
      ],
      elements: [
        {
          id: 1,
          node_i: 1,
          node_j: 2,
          E: 210e9,
          I: 8.5e-6,
          A: 1e-2,
          release_i: false,
          release_j: true
        }
      ],
      point_loads: [
        { node_id: 2, Fx: 0, Fy: -12000, Mz: 2500 }
      ],
      udls: [
        { element_id: 1, w1: -1500, w2: -2500 }
      ],
      element_point_loads: [
        { element_id: 1, a: 1.25, Fx: 3000, Fy: -4000 }
      ]
    });
  });

  it('uses provided axial area and horizontal point load values', async () => {
    const fetchMock = mockFetch({
      ok: true,
      json: vi.fn().mockResolvedValue({
        success: true,
        displacements: [],
        reactions: [],
        internal_forces: []
      })
    });

    await analyzeStructure(
      [
        { id: 1, x: 0, y: 0, support: 'pin' },
        { id: 2, x: 2, y: 0, support: 'roller' }
      ],
      [
        { id: 1, nodeI: 1, nodeJ: 2, E: 200e9, I: 1e-4, A: 2.5e-2, releaseI: true, releaseJ: false }
      ],
      [
        { nodeId: 2, Fx: 6000, Fy: -8000, Mz: 0 }
      ],
      [],
      []
    );

    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(options.body));

    expect(body.elements[0].A).toBe(2.5e-2);
    expect(body.point_loads[0].Fx).toBe(6000);
    expect(body.element_point_loads).toEqual([]);
  });

  it('throws backend detail messages for failed analysis responses', async () => {
    mockFetch({
      ok: false,
      json: vi.fn().mockResolvedValue({ detail: 'Structure is unstable' })
    });

    await expect(analyzeStructure([], [], [], [])).rejects.toThrow('Structure is unstable');
  });
});

describe('healthCheck', () => {
  it('returns true only when the health endpoint responds ok', async () => {
    const fetchMock = mockFetch({ ok: true });

    await expect(healthCheck()).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledWith('http://localhost:8000/health');

    vi.unstubAllGlobals();
    mockFetch({ ok: false });

    await expect(healthCheck()).resolves.toBe(false);
  });
});
