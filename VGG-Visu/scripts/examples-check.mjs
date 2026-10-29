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
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:5173/");
  assert.equal(await page.locator(".example-choice").count(), 9);
  await page.locator("#examples").scrollIntoViewIfNeeded();
  await page.waitForFunction(() =>
    Array.from(document.querySelectorAll("#examples img")).every(
      (i) => i.complete && i.naturalWidth > 0,
    ),
  );
  await page.waitForFunction(() =>
    document.querySelector("#status").textContent.includes("Modèle prêt"),
  );
  await page
    .getByRole("button", { name: "Analyser l’image : Brocoli", exact: true })
    .click();
  await page.waitForFunction(
    () => window.vggDiagnostics().cached.length === 18,
    { timeout: 120000 },
  );
  assert.equal(
    await page.evaluate(() => window.vggDiagnostics().inferences),
    1,
  );
  assert.equal(errors.length, 0);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth),
    390,
  );
  console.log(
    "9 miniatures chargées ; image fournie analysée : 18 sorties réelles ; aucun débordement mobile.",
  );
} finally {
  await browser.close();
}
