import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  await page.goto("http://127.0.0.1:5173/");
  await page.waitForFunction(() =>
    document.querySelector("#status").textContent.includes("Modèle prêt"),
  );
  await page.click("#example");
  await page.waitForFunction(
    () => window.vggDiagnostics().cached.length === 18,
    { timeout: 120000 },
  );
  const before = await page.evaluate(() => window.vggDiagnostics().inferences);
  for (let i = 0; i < 18; i++) {
    await page.locator(`[data-layer="${i}"]`).click();
    assert.equal(
      await page.locator(`[data-layer="${i}"]`).getAttribute("aria-pressed"),
      "true",
    );
    assert.ok(
      (await page.locator("#layer-info").textContent()).includes(
        `Couche ${i + 1} / 18`,
      ),
    );
  }
  await page.locator('[data-layer="17"]').focus();
  await page.keyboard.press("Home");
  assert.equal(
    await page.locator(".network-layer.active").getAttribute("data-layer"),
    "0",
  );
  await page.keyboard.press("ArrowRight");
  assert.equal(
    await page.locator(".network-layer.active").getAttribute("data-layer"),
    "1",
  );
  await page.keyboard.press("End");
  assert.equal(
    await page.locator(".network-layer.active").getAttribute("data-layer"),
    "17",
  );
  assert.equal(
    await page.evaluate(() => window.vggDiagnostics().inferences),
    before,
  );
  await page.locator('[data-layer="7"]').click();
  await page.locator("#network").screenshot({ path: "docs/synoptic.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth),
    390,
  );
  console.log(
    "18 couches sélectionnables ; clavier Home/End/flèche ; aucun recalcul ; aucun débordement mobile.",
  );
} finally {
  await browser.close();
}
