import { chromium } from "@playwright/test";
import fs from "node:fs/promises";
const browser = await chromium.launch({
  executablePath:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: true,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
try {
  const p = await browser.newPage();
  await p.goto("http://127.0.0.1:5173/");
  await p.waitForFunction(() =>
    document.querySelector("#status").textContent.includes("Modèle prêt"),
  );
  const jpg = await p.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 320;
    c.height = 160;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "red";
    ctx.fillRect(0, 0, 320, 160);
    return c.toDataURL("image/jpeg").split(",")[1];
  });
  await p
    .locator("#file")
    .setInputFiles({
      name: "rouge.jpg",
      mimeType: "image/jpeg",
      buffer: Buffer.from(jpg, "base64"),
    });
  await p.waitForFunction(() => window.vggDiagnostics().cached.length === 18, {
    timeout: 120000,
  });
  const jpeg = await p.evaluate(() => ({
    diagnostics: window.vggDiagnostics(),
    pixel: Array.from(
      document
        .querySelector("#input")
        .getContext("2d")
        .getImageData(100, 100, 1, 1).data,
    ),
  }));
  await p
    .locator("#file")
    .setInputFiles({
      name: "invalid.png",
      mimeType: "image/png",
      buffer: Buffer.from("invalid"),
    });
  await p.waitForFunction(() =>
    document.querySelector("#status").textContent.includes("Image illisible"),
  );
  const decodeError = await p.locator("#status").textContent();
  await p.setViewportSize({ width: 390, height: 844 });
  const responsive = await p.evaluate(() => ({
    viewport: innerWidth,
    documentWidth: document.documentElement.scrollWidth,
  }));
  await p.screenshot({ path: "docs/mobile.png", fullPage: true });
  await fs.writeFile(
    "docs/import-results.json",
    JSON.stringify({ jpeg, decodeError, responsive }, null, 2),
  );
  console.log({ decodeError, responsive });
} finally {
  await browser.close();
}
