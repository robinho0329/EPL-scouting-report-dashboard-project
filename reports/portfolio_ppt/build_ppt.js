#!/usr/bin/env node
/**
 * EPL Scout Intelligence — PPT Builder
 * slides/slide-NN.html → Playwright 렌더링 → html2pptx → PPTX 출력
 */

import { createRequire } from "module";
import { readdirSync, existsSync, mkdirSync } from "fs";
import { resolve, dirname, join } from "path";
import { fileURLToPath, pathToFileURL } from "url";

const require = createRequire(import.meta.url);
const __dirname = dirname(fileURLToPath(import.meta.url));

const SLIDES_DIR = join(__dirname, "slides");
const OUTPUT_DIR = join(__dirname, "output");
const HTML2PPTX_PATH = resolve(__dirname, "../../.claude/skills/pptx-skill/scripts/html2pptx.cjs");

// CJS 모듈 직접 require
const html2pptx = require(HTML2PPTX_PATH);

// pptxgenjs도 require로 로드
const pptxgen = require(resolve(__dirname, "../../node_modules/pptxgenjs/dist/pptxgen.cjs.js"));

if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });

const today = new Date().toISOString().slice(0, 10);
const OUTPUT_FILE = join(OUTPUT_DIR, `EPL_Scout_Briefing_${today}.pptx`);

if (!existsSync(SLIDES_DIR)) {
  console.error("❌ slides/ 디렉토리가 없습니다. ppt-designer 에이전트를 먼저 실행하세요.");
  process.exit(1);
}

const slideFiles = readdirSync(SLIDES_DIR)
  .filter(f => f.match(/^slide-\d+\.html$/))
  .sort()
  .map(f => join(SLIDES_DIR, f));

if (slideFiles.length === 0) {
  console.error("❌ slides/ 안에 slide-NN.html 파일이 없습니다.");
  process.exit(1);
}

console.log(`\n📊 EPL Scout Intelligence PPT Builder`);
console.log(`   슬라이드 수: ${slideFiles.length}장`);
console.log(`   출력 경로: ${OUTPUT_FILE}\n`);
slideFiles.forEach((f, i) => console.log(`  [${String(i+1).padStart(2,"0")}] ${f}`));
console.log();

const pptx = new pptxgen();
pptx.layout = "LAYOUT_16x9";

for (const slideUrl of slideFiles) {
  console.log(`  렌더링: ${slideUrl.split("/").pop()}`);
  await html2pptx(slideUrl, pptx);
}

await pptx.writeFile({ fileName: OUTPUT_FILE });
console.log(`\n✅ PPTX 생성 완료: ${OUTPUT_FILE}`);
