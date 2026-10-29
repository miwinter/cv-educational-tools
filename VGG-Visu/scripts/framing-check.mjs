import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage();
  await page.goto("http://127.0.0.1:5173/");
  await page.waitForFunction(() =>
    document.querySelector("#status").textContent.includes("Modèle prêt"),
  );
  assert.equal(await page.locator("#cache").count(), 0);
  const pixel = async (x, y) =>
    page.evaluate(
      ([x, y]) =>
        Array.from(
          document
            .querySelector("#input")
            .getContext("2d")
            .getImageData(x, y, 1, 1).data,
        ),
      [x, y],
    );
  const ready = async (n) =>
    page.waitForFunction(
      (n) =>
        !window.vggDiagnostics().running &&
        window.vggDiagnostics().cached.length === 18 &&
        window.vggDiagnostics().inferences > n,
      n,
      { timeout: 120000 },
    );
  for (const [w, h] of [
    [320, 160],
    [160, 320],
  ]) {
    const png = await page.evaluate(
      ([w, h]) => {
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        ctx.fillStyle = "red";
        ctx.fillRect(0, 0, w, h);
        return c.toDataURL().split(",")[1];
      },
      [w, h],
    );
    await page.selectOption("#framing", "crop");
    let n = await page.evaluate(() => window.vggDiagnostics().inferences);
    await page
      .locator("#file")
      .setInputFiles({
        name: "rectangle.png",
        mimeType: "image/png",
        buffer: Buffer.from(png, "base64"),
      });
    await ready(n);
    assert.deepEqual(await pixel(0, 0), [255, 0, 0, 255]);
    n = await page.evaluate(() => window.vggDiagnostics().inferences);
    await page.selectOption("#framing", "contain");
    await ready(n);
    assert.deepEqual(await pixel(0, 0), [255, 255, 255, 255]);
    assert.deepEqual(await pixel(112, 112), [255, 0, 0, 255]);
    n = await page.evaluate(() => window.vggDiagnostics().inferences);
    await page.selectOption("#framing", "crop");
    await ready(n);
    assert.deepEqual(await pixel(0, 0), [255, 0, 0, 255]);
  }
  console.log(
    "Paysage et portrait : recadrage, marges blanches, retour au recadrage depuis la source originale et 18 sorties vérifiés.",
  );
} finally {
  await browser.close();
}
