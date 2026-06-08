import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { spawn } from 'node:child_process';
import { once } from 'node:events';

const chromePath =
  process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const targetUrl = process.argv[2] ?? 'http://127.0.0.1:5173/?benchmark=1';
const minFps = Number(process.env.MIN_VIEWER_FPS ?? '30');
const timeoutMs = Number(process.env.VIEWER_BENCHMARK_TIMEOUT_MS ?? '30000');
const headless = process.env.VIEWER_BENCHMARK_HEADLESS !== '0';
const userDataDir = await mkdtemp(join(tmpdir(), 'hd-map-viewer-chrome-'));

class CdpClient {
  constructor(webSocketUrl) {
    this.nextId = 1;
    this.pending = new Map();
    this.socket = new WebSocket(webSocketUrl);
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener('open', resolve, { once: true });
      this.socket.addEventListener('error', reject, { once: true });
    });
    this.socket.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }
      this.pending.delete(message.id);
      if (message.error) {
        pending.reject(new Error(JSON.stringify(message.error)));
      } else {
        pending.resolve(message.result);
      }
    });
  }

  send(method, params = {}, sessionId = undefined) {
    const id = this.nextId;
    this.nextId += 1;
    const payload = { id, method, params };
    if (sessionId) {
      payload.sessionId = sessionId;
    }
    this.socket.send(JSON.stringify(payload));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
  }

  close() {
    this.socket.close();
  }
}

async function readDevToolsPort() {
  const portFile = join(userDataDir, 'DevToolsActivePort');
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const [port] = (await readFile(portFile, 'utf-8')).trim().split('\n');
      return port;
    } catch {
      await delay(100);
    }
  }
  throw new Error('Chrome DevTools port was not created before timeout.');
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} returned HTTP ${response.status}`);
  }
  return response.json();
}

const chromeArgs = [
  ...(headless
    ? ['--headless=new', '--use-gl=swiftshader', '--enable-unsafe-swiftshader']
    : []),
  '--window-size=1280,720',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--no-first-run',
  '--no-default-browser-check',
  '--remote-debugging-port=0',
  `--user-data-dir=${userDataDir}`,
  'about:blank',
];

const chrome = spawn(chromePath, chromeArgs);

let cdp;
try {
  const port = await readDevToolsPort();
  const version = await getJson(`http://127.0.0.1:${port}/json/version`);
  cdp = new CdpClient(version.webSocketDebuggerUrl);
  await cdp.open();
  await cdp.send('Browser.getVersion');
  const { targetId } = await cdp.send('Target.createTarget', { url: targetUrl });
  const { sessionId } = await cdp.send('Target.attachToTarget', {
    targetId,
    flatten: true,
  });
  await cdp.send('Runtime.enable', {}, sessionId);

  const deadline = Date.now() + timeoutMs;
  let metrics;
  while (Date.now() < deadline) {
    const result = await cdp.send(
      'Runtime.evaluate',
      {
        expression: 'window.__HD_MAP_VIEWER_METRICS__',
        returnByValue: true,
      },
      sessionId,
    );
    metrics = result.result.value;
    if (metrics?.framesRendered >= metrics?.benchmarkFrameCount) {
      break;
    }
    await delay(250);
  }

  if (!metrics?.averageFps) {
    throw new Error('Viewer metrics were not exposed by the benchmark page.');
  }

  console.log(JSON.stringify(metrics, null, 2));
  if (metrics.averageFps < minFps) {
    process.exit(1);
  }
} finally {
  cdp?.close();
  chrome.kill();
  await Promise.race([once(chrome, 'exit'), delay(2000)]);
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      await rm(userDataDir, { recursive: true, force: true, maxRetries: 3 });
      break;
    } catch (error) {
      if (attempt === 4) {
        console.warn(`Could not remove temporary Chrome profile: ${error.message}`);
      }
      await delay(250);
    }
  }
}
