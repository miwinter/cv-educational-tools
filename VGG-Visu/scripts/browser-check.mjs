import { chromium } from "@playwright/test";
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
const server = http.createServer(async (req, res) => {
  try {
    const name = decodeURIComponent(req.url.split("?")[0]).replace(
      /^\/depot\//,
      "",
    );
    const file = path.join(process.cwd(), "dist", name || "index.html");
    const data = await fs.readFile(file);
    res.setHeader(
      "Content-Type",
      file.endsWith(".js")
        ? "text/javascript"
        : file.endsWith(".css")
          ? "text/css"
          : file.endsWith(".html")
            ? "text/html"
            : file.endsWith(".svg")
              ? "image/svg+xml"
              : file.endsWith(".json")
                ? "application/json"
                : "application/octet-stream",
    );
    res.end(data);
  } catch {
    res.statusCode = 404;
    res.end();
  }
});
await new Promise((r) => server.listen(4175, "127.0.0.1", r));
const browser = await chromium.launch({
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:5173/tests/numerical.html");
  await page.waitForFunction(() => window.runNumerical);
  const numerical = await page.evaluate(() => window.runNumerical());
  await fs.writeFile(
    "docs/numerical-results.json",
    JSON.stringify(numerical, null, 2),
  );
  await page.goto("http://127.0.0.1:5173/");
  await page.waitForFunction(
    () => document.querySelector("#status").textContent.includes("Modèle prêt"),
    { timeout: 120000 },
  );
  const requests = [];
  page.on("request", (r) =>
    requests.push({ url: r.url(), method: r.method() }),
  );
  await page.click("#example");
  await page.waitForFunction(
    () => window.vggDiagnostics().cached.length === 18,
    { timeout: 120000 },
  );
  const first = await page.evaluate(() => window.vggDiagnostics());
  for (let i = 0; i < 18; i++) {
    await page.locator(`[data-layer="${i}"]`).click();
    await page
      .locator("#channel")
      .fill(
        String(
          [
            64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512,
            512, 512, 512, 512,
          ][i],
        ),
      );
    await page.locator("#channel").dispatchEvent("change");
  }
  await page.selectOption("#palette", "heat");
  await page.selectOption("#normalization", "layer");
  const after = await page.evaluate(() => window.vggDiagnostics());
  if (first.inferences !== after.inferences)
    throw Error("Recalcul pendant exploration");
  const memory = [];
  for (let i = 0; i < 4; i++) {
    await page.click("#example");
    await page.waitForFunction(
      (n) =>
        !window.vggDiagnostics().running &&
        window.vggDiagnostics().inferences > n,
      after.inferences + i,
      { timeout: 120000 },
    );
    memory.push(await page.evaluate(() => window.vggDiagnostics().tf));
  }
  await page.screenshot({ path: "docs/application.png", fullPage: true });
  // Real local PNG import, then rapid replacement during inference.
  const png = await page.locator("#input").screenshot();
  await page
    .locator("#file")
    .setInputFiles({ name: "local.png", mimeType: "image/png", buffer: png });
  await page.waitForFunction(() => window.vggDiagnostics().running);
  await page.locator("#file").setInputFiles({
    name: "replacement.png",
    mimeType: "image/png",
    buffer: png,
  });
  const latest = await page.evaluate(() => window.vggDiagnostics().version);
  await page.waitForFunction(
    (v) =>
      !window.vggDiagnostics().running &&
      window.vggDiagnostics().version === v &&
      window.vggDiagnostics().cached.length === 18,
    latest,
    { timeout: 120000 },
  );
  await page.goto("http://127.0.0.1:4175/depot/");
  await page.waitForFunction(
    () => document.querySelector("#status").textContent.includes("Modèle prêt"),
    { timeout: 120000 },
  );
  await page.click("#example");
  await page.waitForFunction(
    () => window.vggDiagnostics().cached.length === 18,
    { timeout: 120000 },
  );
  const subdirectory = await page.evaluate(() => window.vggDiagnostics());
  const report = {
    latest,
    subdirectory,
    first,
    after,
    memory,
    errors,
    requests,
    numericalPass:
      numerical.preprocessingMax === 0 &&
      numerical.results.every((r) => r.failures === 0),
  };
  await fs.writeFile(
    "docs/browser-results.json",
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
  if (errors.length || !report.numericalPass) process.exitCode = 1;
} finally {
  await browser.close();
  server.close();
}
