import { chromium } from '@playwright/test';
import { readFile, stat } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import assert from 'node:assert/strict';
const prefix='/cv-educational-tools/VGG-Visu/';
const server=http.createServer(async(req,res)=>{
 try {
  const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  if(!pathname.startsWith(prefix)) {res.writeHead(404).end();return;}
  let file=path.resolve(pathname.slice(prefix.length)||'index.html');
  if(!file.startsWith(process.cwd()+path.sep))throw Error('Invalid path');
  if((await stat(file)).isDirectory())file=path.join(file,'index.html');
  const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.JPEG':'image/jpeg'};
  res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(await readFile(file));
 } catch {res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
try {
 const page=await browser.newPage();const errors=[],missing=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400&&r.url().includes(prefix))missing.push(r.url());});
 await page.goto(`http://127.0.0.1:${server.address().port}${prefix}`);
 await page.waitForURL('**/site/');
 await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Modèle prêt'));
 assert.equal(await page.locator('.example-choice').count(),9);
 assert.equal(await page.locator('[data-layer]').count(),18);
 assert.equal(await page.locator('#cache').count(),0);
 assert.equal(await page.locator('.input').evaluate(el=>getComputedStyle(el).display),'grid');
 await page.getByRole('button',{name:'Analyser l’image : Brocoli',exact:true}).click();
 await page.waitForFunction(()=>window.vggDiagnostics().cached.length===18,{timeout:120000});
 await page.selectOption('#framing','contain');
 await page.waitForFunction(()=>!window.vggDiagnostics().running&&window.vggDiagnostics().cached.length===18,{timeout:120000});
 assert.deepEqual(errors,[]);assert.deepEqual(missing,[]);
 console.log('Pages statique : redirection, CSS, JS, 9 images, 18 couches, poids et inférence WebGL vérifiés sous '+prefix+'site/. Aucune ressource manquante.');
} finally {await browser.close();server.close();server.closeAllConnections();}
