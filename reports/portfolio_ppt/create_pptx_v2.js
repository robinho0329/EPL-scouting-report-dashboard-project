'use strict';
const PptxGenJS = require('pptxgenjs');

// ─── 디자인 토큰 ───────────────────────────────────────────────────
const D = {
  dark:  '0D1B2A',   // 거의 검정 네이비 (커버)
  navy:  '1E3A5F',   // 네이비 (헤더)
  blue:  '1D4ED8',   // 로열블루 (핵심 강조)
  lblue: 'DBEAFE',   // 연파랑 (카드 배경)
  mblue: '93C5FD',   // 중간 파랑
  slate: '334155',   // 슬레이트 (본문)
  gray:  '64748B',   // 그레이 (캡션)
  lgray: 'F1F5F9',   // 거의 흰색 (카드)
  bord:  'E2E8F0',   // 경계선
  white: 'FFFFFF',
  green: '059669',
  lgreen:'ECFDF5',
  red:   'DC2626',
  lred:  'FEF2F2',
  amber: 'B45309',
  lamber:'FFFBEB',
};

const F = 'Calibri';
// 출력은 이 스크립트와 같은 디렉터리에 만든다 — 절대경로를 박으면 다른 PC에서 못 쓴다
const OUT = require('path').join(__dirname, 'epl_scout_dashboard_portfolio.pptx');

const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE'; // 13.33" × 7.5"
pptx.title  = 'EPL Scout Intelligence Dashboard — Portfolio 2026';

// ─── 공통 헬퍼 ────────────────────────────────────────────────────
// 슬라이드 헤더 (모든 컨텐츠 슬라이드 공통)
function hdr(s, section) {
  s.addShape(pptx.ShapeType.line, { x:0.35,y:0.22,w:12.6,h:0, line:{color:D.bord,width:0.5} });
  s.addText('EPL Scout Intelligence', { x:0.35,y:0.06,w:5,h:0.2, fontSize:8,color:D.gray,fontFace:F,bold:true,charSpacing:1 });
  s.addText(section||'', { x:5,y:0.06,w:4,h:0.2, fontSize:8,color:D.gray,fontFace:F,align:'center' });
  s.addText('Portfolio 2026  ·  Confidential', { x:8.85,y:0.06,w:4.1,h:0.2, fontSize:8,color:D.gray,fontFace:F,align:'right' });
}

// 슬라이드 제목 (두꺼운 좌측 파랑 바 + 제목)
function ttl(s, title, sub) {
  s.addShape(pptx.ShapeType.rect, { x:0.35,y:0.3,w:0.05,h:0.55, fill:{color:D.blue} });
  s.addText(title, { x:0.5,y:0.28,w:12.5,h:0.48, fontSize:21,bold:true,color:D.dark,fontFace:F });
  if(sub) s.addText(sub, { x:0.5,y:0.75,w:12.5,h:0.26, fontSize:10,color:D.gray,fontFace:F });
}

// 푸터 (얇은 선 + 소스 + 페이지)
function ftr(s, n, srcTxt='Source: FBref, Transfermarkt', insight='') {
  s.addShape(pptx.ShapeType.line, { x:0.35,y:6.85,w:12.6,h:0, line:{color:D.bord,width:0.5} });
  if(insight) {
    s.addText('▌ ' + insight, { x:0.35,y:6.88,w:11.5,h:0.25,
      fontSize:8.5,color:D.slate,fontFace:F,italic:true });
  }
  s.addText(srcTxt, { x:0.35,y:7.17,w:10,h:0.2, fontSize:7.5,color:D.gray,fontFace:F });
  s.addText(`${n} / 14`, { x:12.5,y:7.17,w:0.7,h:0.2, fontSize:7.5,color:D.gray,fontFace:F,align:'right' });
}

// 통계 카드 (큰 숫자 + 라벨)
function statCard(s, x, y, w, h, num, label, sub, numColor) {
  s.addShape(pptx.ShapeType.roundRect, { x,y,w,h, fill:{color:D.lgray},line:{color:D.bord,width:0.5},rectRadius:0.03 });
  s.addText(num,   { x,y:y+0.12,w,h:0.62, fontSize:32,bold:true,color:numColor||D.blue,fontFace:F,align:'center' });
  s.addText(label, { x,y:y+0.72,w,h:0.26, fontSize:10,bold:true,color:D.dark,fontFace:F,align:'center' });
  if(sub) s.addText(sub, { x,y:y+0.96,w,h:0.2, fontSize:8.5,color:D.gray,fontFace:F,align:'center' });
}

// 섹션 레이블 (작은 파란 태그)
function tag(s, x, y, txt) {
  s.addShape(pptx.ShapeType.roundRect, { x,y,w:1.5,h:0.22, fill:{color:D.lblue},line:{color:D.mblue,width:0},rectRadius:0.02 });
  s.addText(txt, { x:x+0.05,y:y+0.01,w:1.4,h:0.2, fontSize:8,bold:true,color:D.blue,fontFace:F });
}

// 카드 (흰 배경 + 얇은 테두리)
function card(s, x, y, w, h, bg, border) {
  s.addShape(pptx.ShapeType.roundRect, { x,y,w,h,
    fill:{color:bg||D.lgray}, line:{color:border||D.bord,width:0.6}, rectRadius:0.04 });
}

// 메트릭 행 (라벨 | 값 패턴)
function metricRow(s, x, y, w, label, val, valColor) {
  s.addText(label, { x,y,w:w*0.55,h:0.28, fontSize:9.5,color:D.gray,fontFace:F });
  s.addText(val,   { x:x+w*0.55,y,w:w*0.45,h:0.28, fontSize:9.5,bold:true,color:valColor||D.dark,fontFace:F,align:'right' });
}

// 가로 바 차트 (간단)
function bar(s, x, y, w, h, pct, color) {
  s.addShape(pptx.ShapeType.rect, { x,y,w,h, fill:{color:D.bord} });
  if(pct>0) s.addShape(pptx.ShapeType.rect, { x,y,w:w*Math.min(pct,1),h, fill:{color:color||D.blue} });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 1 — Cover
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  s.background = { color: D.dark };

  // 왼쪽 파란 수직 accent
  s.addShape(pptx.ShapeType.rect, { x:0,y:0,w:0.06,h:7.5, fill:{color:D.blue} });

  // 상단 소문자 브랜드
  s.addText('EPL SCOUT INTELLIGENCE', { x:0.4,y:0.38,w:8,h:0.3,
    fontSize:10,color:D.mblue,fontFace:F,charSpacing:4,bold:false });

  // 메인 타이틀 (두 줄)
  s.addText('Scout Intelligence\nDashboard', { x:0.4,y:0.75,w:8.5,h:2.2,
    fontSize:54,bold:true,color:D.white,fontFace:'Calibri Light',lineSpacingMultiple:1.0 });

  // 구분선
  s.addShape(pptx.ShapeType.line, { x:0.4,y:3.02,w:4.5,h:0, line:{color:D.blue,width:2} });

  // 서브타이틀
  s.addText('FBref + Transfermarkt 기반 EPL 스카우팅 의사결정 지원 시스템', {
    x:0.4,y:3.15,w:11,h:0.38, fontSize:12.5,color:'94A3B8',fontFace:F });

  // 기간
  s.addText('2025.11 — 2026.04', { x:0.4,y:3.6,w:6,h:0.3,
    fontSize:10.5,color:'64748B',fontFace:F });

  // 우측 KPI 패널 (어두운 박스)
  const kpis = [['14종','ML 모델'],['9시즌','EPL 데이터'],['14K+','선수-시즌'],['15P','대시보드']];
  kpis.forEach(([n,l],i) => {
    const x = 9.3 + (i%2)*1.88, y = 2.2 + Math.floor(i/2)*1.7;
    s.addShape(pptx.ShapeType.roundRect, { x,y,w:1.7,h:1.5,
      fill:{color:'152033'}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.05 });
    s.addText(n, { x,y:y+0.25,w:1.7,h:0.72, fontSize:30,bold:true,color:D.white,fontFace:F,align:'center' });
    s.addText(l, { x,y:y+0.95,w:1.7,h:0.3, fontSize:9.5,color:D.mblue,fontFace:F,align:'center' });
  });

  // 하단 기술 스택
  s.addShape(pptx.ShapeType.line, { x:0.4,y:6.55,w:12.5,h:0, line:{color:'1E3A5F',width:0.5} });
  s.addText('Python  ·  XGBoost  ·  LightGBM  ·  Streamlit  ·  Selenium  ·  Optuna  ·  SHAP  ·  pandas', {
    x:0.4,y:6.62,w:12.5,h:0.28, fontSize:9,color:'475569',fontFace:F });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 2 — Project at a Glance
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, 'Overview');
  ttl(s, '데이터 기반 스카우팅 의사결정 시스템', 'FBref·Transfermarkt 크롤링 → ML 14종 → Streamlit 15페이지 대시보드 — end-to-end 구축');

  // KPI row
  const kpis = [
    {n:'14종', l:'ML 모델', s:'예측 p1~p8 + 스카우트 s1~s6'},
    {n:'15P',  l:'대시보드 페이지', s:'Streamlit Cloud 배포'},
    {n:'9',    l:'EPL 시즌', s:'2016/17 ~ 2024/25'},
    {n:'60+',  l:'피처', s:'90분 정규화 기반'},
  ];
  kpis.forEach((k,i) => statCard(s, 0.35+i*3.27, 1.18, 3.08, 1.2, k.n, k.l, k.s));

  // Pipeline
  const steps = [
    {c:D.navy, t:'크롤링', s:'FBref 6초\nTransfermarkt 5초'},
    {c:'1E5F4A',t:'파이프라인', s:'aggregate\npreprocess\nscout_features'},
    {c:'5B2C8E',t:'피처 엔지니어링', s:'90분 정규화\n60+ 피처'},
    {c:'7C2D12',t:'ML 모델 14종', s:'XGBoost·LightGBM\n앙상블·Optuna'},
    {c:'1E3A5F',t:'Streamlit 대시보드', s:'Cloud 배포\n실무 워크플로우'},
  ];
  const sw = 2.26;
  steps.forEach((st,i) => {
    const x = 0.35 + i*(sw+0.18);
    s.addShape(pptx.ShapeType.roundRect, { x,y:2.62,w:sw,h:1.88,
      fill:{color:st.c}, line:{color:D.bord,width:0}, rectRadius:0.04 });
    // dark top band — removed (alpha hex not supported by pptxgenjs)
    s.addText(st.t, { x:x+0.05,y:2.65,w:sw-0.1,h:0.32, fontSize:10,bold:true,color:D.white,fontFace:F,align:'center' });
    s.addText(st.s, { x:x+0.08,y:3.03,w:sw-0.16,h:1.35, fontSize:9,color:'CBD5E1',fontFace:F,align:'center' });
    if(i<4) {
      s.addShape(pptx.ShapeType.line, { x:x+sw+0.03,y:3.5,w:0.14,h:0, line:{color:D.gray,width:1.5} });
    }
  });

  // 모델 성능 미리보기 (6개 태그)
  s.addText('주요 모델 성과', { x:0.35,y:4.68,w:4,h:0.28, fontSize:10,bold:true,color:D.slate,fontFace:F });
  const perf = [
    {code:'S2',t:'시장가치 예측',v:'R² 0.876'},
    {code:'P3',t:'강등 예측',v:'Acc 97.5%'},
    {code:'S3',t:'유사선수 탐색',v:'Sil 0.49'},
    {code:'P8',t:'이적 적응',v:'AUC 0.735'},
    {code:'P1',t:'경기 결과',v:'F1 0.479'},
    {code:'P5',t:'클러스터링',v:'K 포지션별'},
  ];
  perf.forEach((p,i) => {
    const col = i%3, row = Math.floor(i/3);
    const x = 0.35+col*4.35, y = 5.02+row*0.65;
    card(s,x,y,4.15,0.58,D.lgray,D.bord);
    s.addShape(pptx.ShapeType.rect, { x,y,w:0.44,h:0.58, fill:{color:D.navy} });
    s.addText(p.code, { x,y:y+0.01,w:0.44,h:0.56, fontSize:9,bold:true,color:D.white,fontFace:F,align:'center',valign:'middle' });
    s.addText(p.t, { x:x+0.52,y:y+0.08,w:2.2,h:0.28, fontSize:9.5,color:D.slate,fontFace:F });
    s.addText(p.v, { x:x+2.85,y:y+0.06,w:1.2,h:0.3, fontSize:11,bold:true,color:D.blue,fontFace:F,align:'right' });
  });

  ftr(s,2,'Source: FBref, Transfermarkt | 2016/17-2024/25 EPL 9시즌',
    '스카우트 실무자가 즉시 사용 가능한 end-to-end 시스템 — 저평가 탐색·유사선수 대체·이적리스크·강등경보');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 3 — Problem & Solution
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '01. 문제 정의');
  ttl(s, '전통적 스카우팅의 4가지 한계 → 데이터 솔루션', '주관적 평가와 정보 비대칭에서 데이터 기반 의사결정으로 — 김태현 스카우트 페르소나 기반 검증');

  // 헤더 컬럼 라벨
  ['문제 상황','데이터 솔루션','핵심 모델·결과'].forEach((lbl,i) => {
    const x = [0.35,4.75,9.25][i];
    s.addShape(pptx.ShapeType.rect, { x,y:1.12,w:[4.2,4.2,3.95][i],h:0.3, fill:{color:D.dark} });
    s.addText(lbl, { x:x+0.1,y:1.14,w:[4.0,4.0,3.75][i],h:0.26, fontSize:9,bold:true,color:D.white,fontFace:F });
  });

  const rows = [
    {
      q:'Q1. 저평가된 선수를 어떻게 찾나?',
      prob:'시장가치 대비 실제 기여도가 높은 선수를 주관적으로만 발굴 → 탐색 비용 높고 편향 발생',
      sol:'시즌 스탯 + ELO + 팀 의존도 기반 시장가치 예측 → 예측가/실제가 비율로 저평가 자동 탐색',
      model:'S2  XGBoost\nR² = 0.876\nMAE ≈ 3.4M€',
      mc: D.green,
    },
    {
      q:'Q2. 부상·이적 시 대체 선수는?',
      prob:'유사 플레이 스타일 선수 탐색이 전적으로 직관에 의존 → 시간 소요, 오판 위험',
      sol:'90분 정규화 스탯 기반 포지션별 K-means → 같은 클러스터 내 유사 선수 즉시 리스트업',
      model:'S3  K-means\nSilhouette 0.49\n(GK 기준)',
      mc: D.blue,
    },
    {
      q:'Q3. 이적 후 적응 실패 리스크는?',
      prob:'해외→EPL 이적 선수의 적응 예측 불가 → 고액 이적 후 부적응 리스크 상존',
      sol:'source_league + 포지션 + 이전 시즌 G+A 비율로 이진 분류 → 적응 확률 스코어 제공',
      model:'P8  Ensemble\nAUC = 0.735\nRecall 72.2%',
      mc: D.amber,
    },
    {
      q:'Q4. 강등 위험팀을 미리 알 수 있나?',
      prob:'시즌 중반 강등권 판단이 감각에 의존 → 대응 시기 놓침',
      sol:'팀 레벨 30개 피처 (포인트·ELO·시장가치) 기반 XGBoost → 19라운드 기준 조기 경보',
      model:'P3  XGBoost\nAcc = 97.5%\nF1 = 92.3%',
      mc: D.green,
    },
  ];

  rows.forEach((r,i) => {
    const y = 1.52 + i * 1.28;
    // 문제
    card(s, 0.35, y, 4.2, 1.18, D.lgray, D.bord);
    s.addShape(pptx.ShapeType.rect, { x:0.35,y,w:0.04,h:1.18, fill:{color:D.blue} });
    s.addText(r.q, { x:0.48,y:y+0.06,w:3.98,h:0.28, fontSize:9.5,bold:true,color:D.dark,fontFace:F });
    s.addText(r.prob, { x:0.48,y:y+0.35,w:3.98,h:0.76, fontSize:8.5,color:D.slate,fontFace:F,wrap:true });
    // 솔루션
    card(s, 4.75, y, 4.2, 1.18, D.lgray, D.bord);
    s.addText(r.sol, { x:4.9,y:y+0.12,w:3.95,h:0.96, fontSize:8.5,color:D.slate,fontFace:F,wrap:true });
    // 모델
    s.addShape(pptx.ShapeType.roundRect, { x:9.25,y,w:3.95,h:1.18,
      fill:{color:D.dark}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.04 });
    s.addText(r.model, { x:9.35,y:y+0.18,w:3.75,h:0.82,
      fontSize:11,bold:true,color:D.white,fontFace:F,align:'center',valign:'middle' });
  });

  ftr(s,3,'Source: FBref, Transfermarkt',
    '4가지 스카우팅 질문 → 4개 모델 솔루션 — 모두 Streamlit 대시보드에 통합');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 4 — Data Pipeline
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '02. 데이터 수집');
  ttl(s, '체크포인트 기반 크롤링 파이프라인', 'Selenium 자동화 · 레이트 리밋 준수 · 재시작 가능 구조로 9시즌 EPL 데이터 안정적 수집');

  // Source → Raw → Pipeline 3단계 흐름
  // 소스 박스
  const sources = [
    {t:'FBref', sub:'경기별 스탯\n선수별 시즌 스탯\nELO 레이팅 데이터', rate:'요청 간격: 6초'},
    {t:'Transfermarkt', sub:'선수 시장가치\n이적 기록·이적료\n국적·포지션·계약 정보', rate:'요청 간격: 5초'},
  ];
  sources.forEach((src,i) => {
    card(s, 0.35+i*3.85, 1.18, 3.6, 2.55, D.lgray, D.bord);
    s.addShape(pptx.ShapeType.rect, { x:0.35+i*3.85,y:1.18,w:3.6,h:0.42, fill:{color:D.navy} });
    s.addText(src.t, { x:0.45+i*3.85,y:1.22,w:3.4,h:0.34, fontSize:12,bold:true,color:D.white,fontFace:F });
    s.addText(src.sub, { x:0.5+i*3.85,y:1.66,w:3.45,h:1.4, fontSize:9.5,color:D.slate,fontFace:F });
    s.addShape(pptx.ShapeType.roundRect, { x:0.45+i*3.85,y:3.35,w:3.4,h:0.28,
      fill:{color:D.lblue}, line:{color:D.mblue,width:0}, rectRadius:0.02 });
    s.addText('⏱  ' + src.rate, { x:0.5+i*3.85,y:3.37,w:3.3,h:0.24, fontSize:8.5,bold:true,color:D.blue,fontFace:F });
  });

  // 화살표
  s.addShape(pptx.ShapeType.line, { x:7.98,y:2.45,w:0.55,h:0, line:{color:D.gray,width:1.5} });

  // 크롤러 박스
  card(s, 8.55, 1.18, 4.5, 2.55, D.lgray, D.bord);
  s.addShape(pptx.ShapeType.rect, { x:8.55,y:1.18,w:4.5,h:0.42, fill:{color:'5B2C8E'} });
  s.addText('Crawler Engine', { x:8.65,y:1.22,w:4.3,h:0.34, fontSize:12,bold:true,color:D.white,fontFace:F });
  [
    '▸  undetected-chromedriver (봇 감지 우회)',
    '▸  체크포인트 DB — 중단 지점 재개',
    '▸  시즌 단위 순차 수집 (2016/17~)',
    '▸  팀명 fuzzy matching 표준화',
    '▸  JSON → CSV 자동 변환 저장',
  ].forEach((t,i) => {
    s.addText(t, { x:8.65,y:1.66+i*0.38,w:4.3,h:0.32, fontSize:9.5,color:D.slate,fontFace:F });
  });

  // 데이터 흐름도 (하단)
  s.addShape(pptx.ShapeType.line, { x:0.35,y:3.9,w:12.6,h:0, line:{color:D.bord,width:0.5} });
  s.addText('수집된 데이터 저장 구조', { x:0.35,y:3.98,w:6,h:0.28, fontSize:10,bold:true,color:D.slate,fontFace:F });

  const folders = [
    {f:'data/raw/matches/', d:'경기별 스탯 CSV\n38R × 9시즌'},
    {f:'data/raw/players/', d:'선수별 시즌 스탯\n14,000+ rows'},
    {f:'data/raw/market_value/', d:'Transfermarkt\n시즌별 시장가치'},
    {f:'data/raw/transfers/', d:'이적 기록\n이적료·리그 정보'},
  ];
  folders.forEach((fd,i) => {
    card(s, 0.35+i*3.27, 4.32, 3.08, 1.0, 'FAFBFF', D.bord);
    s.addShape(pptx.ShapeType.rect, { x:0.35+i*3.27,y:4.32,w:0.04,h:1.0, fill:{color:D.blue} });
    s.addText(fd.f, { x:0.48+i*3.27,y:4.38,w:2.85,h:0.26, fontSize:9,bold:true,color:D.blue,fontFace:F });
    s.addText(fd.d, { x:0.48+i*3.27,y:4.65,w:2.85,h:0.6, fontSize:9,color:D.slate,fontFace:F });
  });

  // 체크포인트 하이라이트
  s.addShape(pptx.ShapeType.roundRect, { x:0.35,y:5.52,w:12.6,h:0.65,
    fill:{color:D.lblue}, line:{color:D.mblue,width:0.5}, rectRadius:0.03 });
  s.addText('🔄  체크포인트 재시작  —  SQLite 기반 진행률 DB로 네트워크 오류·차단 발생 시 마지막 수집 지점부터 자동 재개. 대용량 멀티시즌 수집의 핵심 안정화 장치', {
    x:0.5,y:5.58,w:12.3,h:0.52, fontSize:10,color:D.navy,fontFace:F,valign:'middle' });

  ftr(s,4,'Source: FBref.com, Transfermarkt.com',
    'FBref 6초·TM 5초 레이트 리밋 + 체크포인트로 9시즌 14,000+ rows 무중단 수집 완료');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 5 — Dataset & Feature Engineering
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '03. 데이터 · 피처');
  ttl(s, '9시즌 통합 데이터셋 + 60+ 피처 — 포지션 중립 평가 체계', '시간 기반 Train/Val/Test 분리로 데이터 누수 차단 · 90분 정규화로 출전 시간 불문 공정 비교');

  // 좌: 데이터 현황
  card(s,0.35,1.18,6.1,2.45,D.lgray,D.bord);
  s.addText('데이터셋 현황', { x:0.5,y:1.25,w:5.8,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });

  const dstats = [
    {l:'전체 경기',v:'3,420+',s:'EPL 매치 레벨'},
    {l:'선수-시즌 Rows',v:'14,000+',s:'ML 학습 단위'},
    {l:'Train / Val / Test',v:'8,800 / 1,210 / 1,234',s:'시간 순 분리'},
    {l:'피처 수 (S2 기준)',v:'43개',s:'90분 정규화 포함'},
    {l:'스카우팅 대상',v:'617명',s:'900분 이상 출전'},
  ];
  dstats.forEach((d,i) => {
    s.addShape(pptx.ShapeType.line, { x:0.45,y:1.6+i*0.38,w:5.9,h:0, line:{color:D.bord,width:0.4} });
    s.addText(d.l, { x:0.48,y:1.62+i*0.38,w:2.8,h:0.3, fontSize:9.5,color:D.gray,fontFace:F });
    s.addText(d.v, { x:3.2,y:1.62+i*0.38,w:2.2,h:0.3, fontSize:9.5,bold:true,color:D.dark,fontFace:F,align:'right' });
    if(d.s) s.addText(d.s, { x:0.48,y:1.62+i*0.38,w:5.9,h:0.28, fontSize:8,color:D.gray,fontFace:F,align:'right' });
  });

  // Train/Val/Test 시각 바
  s.addText('시간 순 데이터 분리', { x:0.5,y:3.7,w:5.8,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  const splits = [{l:'Train',s:'2016~2021',w:3.9,c:D.navy},{l:'Val',s:'21~23',w:1.05,c:D.blue},{l:'Test',s:'23~25',w:1.05,c:D.mblue}];
  let sx=0.45;
  splits.forEach(sp => {
    s.addShape(pptx.ShapeType.rect, { x:sx,y:4.04,w:sp.w,h:0.45, fill:{color:sp.c} });
    s.addText(sp.l, { x:sx,y:4.06,w:sp.w,h:0.2, fontSize:8.5,bold:true,color:D.white,fontFace:F,align:'center' });
    s.addText(sp.s, { x:sx,y:4.25,w:sp.w,h:0.2, fontSize:7.5,color:'CBD5E1',fontFace:F,align:'center' });
    sx += sp.w + 0.04;
  });

  // 우: 피처 카테고리
  card(s,6.65,1.18,6.3,4.35,D.lgray,D.bord);
  s.addText('피처 엔지니어링 카테고리', { x:6.8,y:1.25,w:6.0,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });

  const cats = [
    {c:'C0392B',t:'공격',items:'goals_p90 · assists_p90 · xG_p90 · gc_p90'},
    {c:D.blue, t:'수비',items:'tackles_p90 · clearances_p90 · blocks_p90'},
    {c:'059669',t:'경험·이적',items:'epl_experience · transfer_count · is_cross_league'},
    {c:'7C3AED',t:'팀 강도',items:'ELO(avg+last) · rolling form 3/5/10 · ppg'},
    {c:'D97706',t:'시장가치',items:'log_mv_prev · mv_change_pct · age_premium'},
    {c:D.slate, t:'선수 프로파일',items:'age_sq · height · is_international · war_norm'},
  ];
  cats.forEach((c,i) => {
    const y = 1.62 + i * 0.6;
    s.addShape(pptx.ShapeType.roundRect, { x:6.78,y,w:0.28,h:0.28, fill:{color:c.c},rectRadius:0.02,line:{color:D.bord,width:0} });
    s.addText(c.t, { x:7.12,y,w:1.2,h:0.28, fontSize:9.5,bold:true,color:D.dark,fontFace:F });
    s.addText(c.items, { x:8.32,y,w:4.55,h:0.28, fontSize:9,color:D.gray,fontFace:F });
    if(i<5) s.addShape(pptx.ShapeType.line, { x:6.78,y:y+0.32,w:6.1,h:0, line:{color:D.bord,width:0.4} });
  });

  // 핵심 원칙 박스
  card(s,6.78,5.27,6.1,0.65,'EFF6FF',D.mblue);
  s.addText('설계 원칙  —  과거 시즌 데이터만 사용(누수 차단) · 포지션별 가중치 분리 · 90분 정규화로 출전 시간 무관 공정 비교', {
    x:6.92,y:5.33,w:5.82,h:0.52, fontSize:9,color:D.navy,fontFace:F,valign:'middle',wrap:true });

  ftr(s,5,'Source: FBref, Transfermarkt | 파이프라인: aggregate → preprocess → scout_features',
    '시간 순 분리 + 90분 정규화 — 데이터 누수 없는 엄격한 ML 실험 설계');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 6 — S2 Market Value Model
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '04-A. 모델: S2 시장가치 예측');
  ttl(s, 'XGBoost 시장가치 예측으로 저평가 선수 자동 탐색', '43개 피처 기반 | 예측가/실제가 비율 ≥1.5× → 저평가 | 스마트 필터 v4 적용');

  // Hero metric
  s.addShape(pptx.ShapeType.roundRect, { x:0.35,y:1.18,w:2.65,h:2.55,
    fill:{color:D.dark}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.05 });
  s.addText('R²', { x:0.35,y:1.3,w:2.65,h:0.4, fontSize:12,color:D.mblue,fontFace:F,align:'center' });
  s.addText('0.876', { x:0.35,y:1.65,w:2.65,h:0.88, fontSize:48,bold:true,color:D.white,fontFace:F,align:'center' });
  s.addShape(pptx.ShapeType.line, { x:0.55,y:2.58,w:2.25,h:0, line:{color:'1E3A5F',width:0.5} });
  s.addText('MAE  3.4M €', { x:0.35,y:2.64,w:2.65,h:0.28, fontSize:9.5,bold:true,color:D.mblue,fontFace:F,align:'center' });
  s.addText('MAPE  29.8%', { x:0.35,y:2.92,w:2.65,h:0.28, fontSize:9.5,color:'64748B',fontFace:F,align:'center' });
  s.addText('Train 8,800\nTest 1,234', { x:0.35,y:3.25,w:2.65,h:0.4, fontSize:8.5,color:'475569',fontFace:F,align:'center' });

  // 저평가 Top 6 테이블
  card(s, 3.18, 1.18, 4.72, 2.55, D.lgray, D.bord);
  s.addShape(pptx.ShapeType.rect, { x:3.18,y:1.18,w:4.72,h:0.38, fill:{color:D.green} });
  s.addText('저평가 Top 선수 (예측가 / 실제가)', { x:3.28,y:1.22,w:4.52,h:0.3, fontSize:10,bold:true,color:D.white,fontFace:F });

  const unders = [
    ['Oliver Arblaster','Sheffield Utd · MF','450K','2.97M','6.61×'],
    ['Sam Morsy','Ipswich · MF','500K','2.16M','4.33×'],
    ['Jakub Stolarczyk','Leicester · GK','1.0M','3.83M','3.83×'],
    ['Alex Palmer','Ipswich · GK','1.0M','2.54M','2.54×'],
    ['Cameron Burgess','Ipswich · DF','2.0M','3.76M','1.88×'],
  ];
  // header row
  ['선수','팀·포지션','실제가','예측가','비율'].forEach((h,j) => {
    const xs=[3.22,4.72,6.12,6.82,7.52][j], ws=[1.45,1.35,0.65,0.65,0.78][j];
    s.addText(h, { x:xs,y:1.62,w:ws,h:0.25, fontSize:8,bold:true,color:D.gray,fontFace:F });
  });
  unders.forEach((r,i) => {
    if(i%2===0) s.addShape(pptx.ShapeType.rect, { x:3.22,y:1.88+i*0.33,w:4.6,h:0.33, fill:{color:'F0FFF4'} });
    [r[0],r[1],r[2],r[3],r[4]].forEach((v,j) => {
      const xs=[3.22,4.72,6.12,6.82,7.52][j], ws=[1.45,1.35,0.65,0.65,0.78][j];
      const isBold = j===4;
      const clr = j===4 ? D.green : (j>=2 ? D.slate : D.dark);
      s.addText(v, { x:xs,y:1.9+i*0.33,w:ws,h:0.28, fontSize:9,bold:isBold,color:clr,fontFace:F });
    });
  });

  // 과대평가 Top 3 테이블
  card(s, 8.1, 1.18, 5.1, 2.55, D.lgray, D.bord);
  s.addShape(pptx.ShapeType.rect, { x:8.1,y:1.18,w:5.1,h:0.38, fill:{color:D.red} });
  s.addText('과대평가 주의 선수', { x:8.2,y:1.22,w:4.9,h:0.3, fontSize:10,bold:true,color:D.white,fontFace:F });

  const overs = [
    ['Jacob Greaves','Ipswich DF','18M €','3.78M €','0.21×'],
    ['Ibrahim Sangaré',"Nott'm MF",'30M €','6.60M €','0.22×'],
    ['Manuel Ugarte','Man Utd MF','45M €','10.5M €','0.23×'],
    ['Matthijs de Ligt','Man Utd DF','38M €','11.6M €','0.30×'],
    ['Joshua Zirkzee','Man Utd FW','30M €','10.4M €','0.35×'],
  ];
  ['선수','팀·포지션','시장가','예측가','비율'].forEach((h,j) => {
    const xs=[8.15,9.55,10.75,11.45,12.15][j], ws=[1.35,1.15,0.65,0.65,0.88][j];
    s.addText(h, { x:xs,y:1.62,w:ws,h:0.25, fontSize:8,bold:true,color:D.gray,fontFace:F });
  });
  overs.forEach((r,i) => {
    if(i%2===0) s.addShape(pptx.ShapeType.rect, { x:8.15,y:1.88+i*0.33,w:4.95,h:0.33, fill:{color:'FFF5F5'} });
    r.forEach((v,j) => {
      const xs=[8.15,9.55,10.75,11.45,12.15][j], ws=[1.35,1.15,0.65,0.65,0.88][j];
      s.addText(v, { x:xs,y:1.9+i*0.33,w:ws,h:0.28, fontSize:9,bold:j===4,color:j===4?D.red:D.dark,fontFace:F });
    });
  });

  // 하단: 필터 설명 + 피처 중요도
  card(s, 0.35, 3.88, 6.1, 1.35, D.lgray, D.bord);
  s.addText('v4 스마트 필터', { x:0.5,y:3.95,w:5.8,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  [
    {c:D.green,t:'저평가 목록: 38세+ 제외 — 나이 감가상각 ≠ 저평가 신호'},
    {c:D.red,  t:'과대평가 목록: 21세 이하·(22세 이하+1,500분 미만) 유스 잠재력 보정'},
    {c:D.blue, t:'스카우팅 대상: 900분 이상 출전 617명 최종 산출'},
  ].forEach((f,i) => {
    s.addShape(pptx.ShapeType.ellipse, { x:0.48,y:4.31+i*0.3,w:0.1,h:0.1, fill:{color:f.c} });
    s.addText(f.t, { x:0.64,y:4.27+i*0.3,w:5.7,h:0.28, fontSize:9,color:D.slate,fontFace:F });
  });

  card(s, 6.65, 3.88, 6.3, 1.35, D.lgray, D.bord);
  s.addText('XGBoost 피처 중요도 Top 5', { x:6.8,y:3.95,w:6.0,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  [['n_matches (출전 경기수)',0.169],['log_mv_prev (전시즌 가치)',0.110],['war_norm',0.064],['win_rate_with_player',0.059],['consistency_mean',0.058]].forEach(([f,v],i)=>{
    s.addText(f, { x:6.8,y:4.31+i*0.24,w:3.5,h:0.22, fontSize:8.5,color:D.slate,fontFace:F });
    bar(s,10.1,4.34+i*0.24,2.5,0.16,v/0.169,D.navy);
    s.addText((v*100).toFixed(1)+'%', { x:12.65,y:4.31+i*0.24,w:0.65,h:0.22, fontSize:8.5,color:D.gray,fontFace:F });
  });

  ftr(s,6,'Source: Transfermarkt 시장가치 + FBref 스탯 | XGBoost v4',
    '예측가/실제가 비율로 저평가 자동 탐색 — S2 저평가 탐색기 대시보드에서 포지션·비율 필터로 즉시 조회');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 7 — S3 Clustering & P5
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '04-B. 모델: S3 포지션별 클러스터링');
  ttl(s, '포지션별 K-means — 유사 선수 탐색 · 대체 후보 즉시 리스트업', 'Silhouette 0.115 (전체 혼합) → 포지션별 분리 후 0.21~0.49 대폭 개선');

  // Before → After
  card(s, 0.35, 1.18, 4.1, 2.0, D.lred, D.red);
  s.addShape(pptx.ShapeType.rect, { x:0.35,y:1.18,w:4.1,h:0.35, fill:{color:D.red} });
  s.addText('Before  —  전체 통합 K=6', { x:0.45,y:1.21,w:3.9,h:0.28, fontSize:10,bold:true,color:D.white,fontFace:F });
  s.addText('Silhouette Score', { x:0.45,y:1.62,w:2.5,h:0.28, fontSize:10,color:D.slate,fontFace:F });
  s.addText('0.115', { x:2.9,y:1.52,w:1.4,h:0.48, fontSize:32,bold:true,color:D.red,fontFace:F });
  s.addText('FW · MF · DF · GK 혼합\n→ 포지션 섞임으로 군집 품질 저하\n→ 유사 선수 탐색 신뢰도 낮음', { x:0.45,y:2.05,w:3.9,h:0.88, fontSize:9,color:D.slate,fontFace:F });

  s.addText('→', { x:4.55,y:1.95,w:0.5,h:0.5, fontSize:26,bold:true,color:D.blue,fontFace:F,align:'center',valign:'middle' });

  card(s, 5.15, 1.18, 4.1, 2.0, D.lgreen, D.green);
  s.addShape(pptx.ShapeType.rect, { x:5.15,y:1.18,w:4.1,h:0.35, fill:{color:D.green} });
  s.addText('After  —  포지션별 분리', { x:5.25,y:1.21,w:3.9,h:0.28, fontSize:10,bold:true,color:D.white,fontFace:F });
  s.addText('Silhouette (GK)', { x:5.25,y:1.62,w:2.5,h:0.28, fontSize:10,color:D.slate,fontFace:F });
  s.addText('0.487', { x:7.62,y:1.52,w:1.4,h:0.48, fontSize:32,bold:true,color:D.green,fontFace:F });
  s.addText('포지션별 최적 K 선택\n→ 포지션 내 스타일 순수 비교\n→ 유사 선수 탐색 정밀도 4배 향상', { x:5.25,y:2.05,w:3.9,h:0.88, fontSize:9,color:D.slate,fontFace:F });

  // 포지션별 결과 표
  card(s, 9.45, 1.18, 3.8, 2.0, D.lgray, D.bord);
  s.addText('포지션별 최적 K & Silhouette', { x:9.55,y:1.25,w:3.6,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  s.addShape(pptx.ShapeType.rect, { x:9.5,y:1.58,w:3.7,h:0.3, fill:{color:D.dark} });
  ['포지션','K','Silhouette'].forEach((h,j) => {
    s.addText(h, { x:[9.53,10.45,11.35][j],y:1.6,w:[0.85,0.85,1.08][j],h:0.26, fontSize:8.5,bold:true,color:D.white,fontFace:F });
  });
  [['FW','7','0.2085'],['MID','7','0.2299'],['DEF','4','0.3637'],['GK','2','0.4873']].forEach(([pos,k,sil],i) => {
    s.addShape(pptx.ShapeType.rect, { x:9.5,y:1.88+i*0.3,w:3.7,h:0.3, fill:{color:i%2===0?'F8FAFC':D.white} });
    s.addText(pos, { x:9.55,y:1.91+i*0.3,w:0.85,h:0.24, fontSize:9,bold:true,color:D.navy,fontFace:F });
    s.addText(k,   { x:10.45,y:1.91+i*0.3,w:0.85,h:0.24, fontSize:9,color:D.slate,fontFace:F,align:'center' });
    s.addText(sil, { x:11.35,y:1.91+i*0.3,w:1.08,h:0.24, fontSize:9,bold:true,color:D.green,fontFace:F,align:'right' });
  });

  // 활용 시나리오 3개
  s.addShape(pptx.ShapeType.line, { x:0.35,y:3.38,w:12.6,h:0, line:{color:D.bord,width:0.5} });
  s.addText('스카우팅 활용 시나리오', { x:0.35,y:3.45,w:8,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });

  const cases = [
    {ic:'🔍',t:'유사 선수 즉시 탐색',d:'특정 선수 선택 → 같은 클러스터 내\n스타일 유사 선수 즉시 리스트업\n대시보드 "선수 유형 탐색기" 실시간 제공'},
    {ic:'🔄',t:'부상·이적 대체 시나리오',d:'주전 부상 시 동일 클러스터에서\n대체 후보 즉시 탐색\n예산·나이·리그 조건 필터 가능'},
    {ic:'📊',t:'포지션 유형 분류',d:'FW 7유형 (박스·윙·딥라잉 등)\nMID 7유형 (박스투박스·수비형 등)\nDEF 4유형 · GK 2유형 자동 분류'},
  ];
  cases.forEach((c,i) => {
    card(s, 0.35+i*4.3, 3.8, 4.1, 2.42, D.lgray, D.bord);
    s.addText(`${c.ic}  ${c.t}`, { x:0.5+i*4.3,y:3.88,w:3.8,h:0.32, fontSize:10,bold:true,color:D.dark,fontFace:F });
    s.addShape(pptx.ShapeType.line, { x:0.5+i*4.3,y:4.22,w:3.8,h:0, line:{color:D.bord,width:0.5} });
    s.addText(c.d, { x:0.5+i*4.3,y:4.28,w:3.8,h:1.82, fontSize:9.5,color:D.slate,fontFace:F });
  });

  ftr(s,7,'Source: FBref 선수 스탯 | S3 v2 포지션별 분리 모델',
    '포지션 섞임 해소 → Silhouette 4배 향상 — 유사 선수 탐색 신뢰도 실무 활용 가능 수준 달성');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 8 — P8 Transfer Adaptation
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '04-C. 모델: P8 이적 적응 예측');
  ttl(s, '이적 적응 리스크 예측 — 회귀 실패에서 이진 분류 전환으로 AUC 0.735', '타겟: 이적 후 G+A/90 이전 시즌 80% 유지 여부 | 500분+ 필터 | XGB+LR+RF 앙상블');

  // Hero
  s.addShape(pptx.ShapeType.roundRect, { x:0.35,y:1.18,w:2.45,h:2.42,
    fill:{color:D.dark}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.05 });
  s.addText('AUC', { x:0.35,y:1.32,w:2.45,h:0.3, fontSize:11,color:D.mblue,fontFace:F,align:'center' });
  s.addText('0.735', { x:0.35,y:1.6,w:2.45,h:0.75, fontSize:44,bold:true,color:D.white,fontFace:F,align:'center' });
  s.addShape(pptx.ShapeType.line, { x:0.5,y:2.38,w:2.15,h:0, line:{color:'1E3A5F',width:0.5} });
  [['F1','0.669'],['Recall','0.722'],['Acc','0.683']].forEach(([k,v],i) => {
    s.addText(k+': '+v, { x:0.35,y:2.44+i*0.3,w:2.45,h:0.28, fontSize:9.5,bold:true,color:D.mblue,fontFace:F,align:'center' });
  });

  // Pivot 스토리
  card(s, 2.95, 1.18, 9.7, 1.12, D.lgray, D.bord);
  s.addShape(pptx.ShapeType.rect, { x:2.95,y:1.18,w:0.04,h:1.12, fill:{color:D.amber} });
  s.addText('모델 방향 전환 스토리', { x:3.05,y:1.24,w:9.4,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  s.addText('회귀 4일 연속 하락: R² 0.1736 → 0.1572 → 0.1269   →   미팅 결정: "연속 수치 예측 불가, 이진 분류로 전환"   →   타겟 재정의 + 앙상블로 AUC 0.735 달성', {
    x:3.05,y:1.55,w:9.4,h:0.68, fontSize:9.5,color:D.slate,fontFace:F,valign:'middle' });

  // 리스크 분류 3열
  const risks = [
    {l:'High Risk',d:'adapt_proba ≤ 0.40',n:'514명',pct:0.363,c:D.red,bg:D.lred,desc:'영입 신중\n계약 리스크 경고'},
    {l:'Medium',d:'0.40 < proba < 0.70',n:'649명',pct:0.459,c:D.amber,bg:D.lamber,desc:'추가 스카우팅\n검증 필요'},
    {l:'Low Risk',d:'adapt_proba ≥ 0.70',n:'253명',pct:0.179,c:D.green,bg:D.lgreen,desc:'영입 추천\n적응 성공 가능'},
  ];
  risks.forEach((r,i) => {
    const x = 2.95 + i * 3.28;
    card(s, x, 2.45, 3.08, 2.32, r.bg, r.c);
    s.addText(r.l, { x:x+0.1,y:2.52,w:2.88,h:0.3, fontSize:11,bold:true,color:r.c,fontFace:F });
    s.addText(r.d, { x:x+0.1,y:2.82,w:2.88,h:0.26, fontSize:8.5,color:D.slate,fontFace:F });
    s.addText(r.n, { x:x+0.1,y:3.08,w:2.88,h:0.6, fontSize:36,bold:true,color:r.c,fontFace:F,align:'center' });
    s.addText(r.desc, { x:x+0.1,y:3.68,w:2.88,h:0.9, fontSize:9,color:D.slate,fontFace:F,align:'center' });
  });

  // 피처 중요도 + 타겟 정의
  card(s, 0.35, 4.9, 5.95, 1.35, D.lgray, D.bord);
  s.addText('Top 피처 중요도 (XGBoost)', { x:0.5,y:4.97,w:5.7,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  [['포지션 코드 (pos_code)',0.0727],['G+A/90 전시즌 비율',0.0588],['출신 리그: La Liga',0.0532],['G+A/90 이전 시즌',0.0506],['팀 포인트 차이',0.0497]].forEach(([f,v],i) => {
    s.addText(f, { x:0.5,y:5.3+i*0.2,w:3.0,h:0.2, fontSize:8.5,color:D.slate,fontFace:F });
    bar(s, 3.55, 5.33+i*0.2, 2.5, 0.15, v/0.0727, D.navy);
  });

  card(s, 6.5, 4.9, 6.7, 1.35, D.lgray, D.bord);
  s.addShape(pptx.ShapeType.rect, { x:6.5,y:4.9,w:0.04,h:1.35, fill:{color:D.blue} });
  s.addText('타겟 정의 & 앙상블', { x:6.62,y:4.97,w:6.4,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  [
    'adapted = 1  if  G+A/90_new / G+A/90_old  ≥  0.80',
    '(G+A/90_old = 0 이면  G+A/90_new > 0.10)',
    '최소 출전 필터: 90s_new ≥ 5.5 (≈500분)',
    '앙상블: XGBoost + LogisticRegression + RandomForest (소프트 보팅)',
    'class_weight = balanced (적응 성공/실패 44:56 불균형 보정)',
  ].forEach((t,i) => {
    s.addText(t, { x:6.62,y:5.3+i*0.2,w:6.4,h:0.2, fontSize:8.5,color:D.slate,fontFace:F });
  });

  ftr(s,8,'Source: FBref | 학습 1,132건 / 테스트 284건',
    'source_league + 포지션이 핵심 피처 — 리그 레벨 차이와 포지션별 G+A 유지율이 적응 성공의 핵심 예측 변수');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 9 — P1 & P3 Prediction Models
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '04-D. 모델: P1 경기 결과 · P3 강등 예측');
  ttl(s, 'P1 경기 결과 예측 (F1 +18.7%) · P3 강등 조기 경보 (Acc 97.5%)', 'P1: ELO + H2H + 롤링 폼 88 피처 | P3: 팀 레벨 30 피처 Full/Mid-season 이중 모드');

  // P1 섹션
  tag(s, 0.35, 1.18, 'P1 경기 결과');
  // Before/After
  card(s,0.35,1.47,3.28,2.7,D.lgray,D.bord);
  s.addText('Baseline v1', { x:0.5,y:1.55,w:2.9,h:0.28, fontSize:9.5,color:D.gray,fontFace:F,align:'center' });
  [['F1 Macro','0.404'],['Accuracy','53.8%'],['무승부 Recall','0%'],['피처','49개']].forEach(([k,v],i) => {
    metricRow(s,0.48,1.88+i*0.48,3.0,k,v,D.gray);
  });

  s.addText('+18.7%\n→', { x:3.72,y:2.1,w:0.72,h:0.75, fontSize:11,bold:true,color:D.green,fontFace:F,align:'center',valign:'middle' });

  card(s,4.55,1.47,3.28,2.7,'EFF6FF',D.blue);
  s.addShape(pptx.ShapeType.rect, { x:4.55,y:1.47,w:3.28,h:0.32, fill:{color:D.blue} });
  s.addText('v2 개선판', { x:4.65,y:1.49,w:3.08,h:0.26, fontSize:9.5,bold:true,color:D.white,fontFace:F,align:'center' });
  [['F1 Macro','0.479'],['Accuracy','52.3%'],['무승부 Recall','22%'],['피처','88개']].forEach(([k,v],i) => {
    metricRow(s,4.65,1.88+i*0.48,3.0,k,v,D.dark);
    s.addShape(pptx.ShapeType.roundRect, { x:4.65,y:1.91+i*0.48,w:3.0,h:0.28,
      fill:{color:D.lblue}, line:{color:D.bord,width:0}, rectRadius:0.02 });
    metricRow(s,4.65,1.88+i*0.48,3.0,k,v,D.dark);
  });

  // P1 피처 중요도 bar
  card(s,0.35,4.28,7.58,1.95,D.lgray,D.bord);
  s.addText('Top 피처 중요도', { x:0.5,y:4.35,w:3,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  s.addText('ELO diff가 전체의 11%로 압도적 1위 — 팀 실력 차이가 결과 예측의 핵심', { x:0.5,y:4.62,w:7.3,h:0.25, fontSize:8.5,color:D.gray,fontFace:F });
  [['elo_diff',0.109,'압도적 1위'],['elo_ratio',0.071,''],['elo_diff_abs',0.031,''],['home_defense_5',0.018,''],['season_gd_diff',0.016,'']].forEach(([f,v,note],i) => {
    s.addText(f, { x:0.5,y:4.93+i*0.22,w:2.7,h:0.2, fontSize:8.5,color:D.slate,fontFace:F });
    bar(s,3.25,4.96+i*0.22,4.25,0.16,v/0.109,i===0?D.dark:D.mblue);
    if(note) s.addText(note, { x:7.55,y:4.93+i*0.22,w:0.9,h:0.2, fontSize:7.5,color:D.blue,fontFace:F });
  });

  // P3 섹션 구분선
  s.addShape(pptx.ShapeType.line, { x:8.15,y:1.15,w:0,h:5.68, line:{color:D.bord,width:0.6} });
  tag(s, 8.3, 1.18, 'P3 강등 예측');

  // P3 Hero
  s.addShape(pptx.ShapeType.roundRect, { x:8.3,y:1.47,w:4.88,h:1.22,
    fill:{color:D.dark}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.04 });
  s.addText('97.5%', { x:8.3,y:1.52,w:2.3,h:0.98, fontSize:46,bold:true,color:D.white,fontFace:F,align:'center' });
  s.addText('Full-season\nAccuracy', { x:10.62,y:1.6,w:1.5,h:0.55, fontSize:10,color:D.mblue,fontFace:F });
  s.addText('F1 92.3%', { x:10.62,y:2.1,w:2.4,h:0.32, fontSize:12,bold:true,color:D.mblue,fontFace:F });

  // P3 모드 비교
  card(s,8.3,2.82,4.88,1.3,D.lgray,D.bord);
  s.addText('두 가지 예측 모드', { x:8.45,y:2.89,w:4.6,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  s.addShape(pptx.ShapeType.rect, { x:8.38,y:3.22,w:4.72,h:0.3, fill:{color:D.dark} });
  ['모드','Test Acc','F1','Val Acc'].forEach((h,j) => {
    s.addText(h, { x:[8.42,9.52,10.52,11.32][j],y:3.25,w:[1.05,0.95,0.75,0.85][j],h:0.24, fontSize:8,bold:true,color:D.white,fontFace:F });
  });
  [['Full-season','97.5%','92.3%','97.5%'],['Mid-season (19R)','100%','100%','90.0%']].forEach((r,i) => {
    s.addShape(pptx.ShapeType.rect, { x:8.38,y:3.52+i*0.32,w:4.72,h:0.32, fill:{color:i%2===0?'F0F4F8':D.white} });
    r.forEach((v,j) => {
      s.addText(v, { x:[8.42,9.52,10.52,11.32][j],y:3.55+i*0.32,w:[1.05,0.95,0.75,0.85][j],h:0.26,
        fontSize:9,color:j>0?D.blue:D.dark,bold:j>0,fontFace:F });
    });
  });

  // P3 피처
  card(s,8.3,4.28,4.88,1.95,D.lgray,D.bord);
  s.addText('핵심 피처 (30개)', { x:8.45,y:4.35,w:4.6,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  ['points · ppg · win_rate','goal_diff · goals_for/against',
   'ELO (avg + last)  ·  form_5','squad 시장가치 · avg_epl_exp','promoted flag · shots_on_target'].forEach((t,i) => {
    s.addText('▸  '+t, { x:8.45,y:4.65+i*0.3,w:4.6,h:0.28, fontSize:9,color:D.slate,fontFace:F });
  });

  ftr(s,9,'Source: FBref | P1: Train 7,890/Val 760/Test 730 | P3: Train 2000-2021/Test 2023-2025',
    'P1: ELO diff 압도적 1위 — 팀 실력 차이가 결과 예측의 핵심. P3: 팀 레벨 피처만으로 Acc 97.5% 달성');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 10 — Streamlit Dashboard
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '05. Streamlit 대시보드');
  ttl(s, '15페이지 Streamlit 대시보드 — 스카우트 실무 워크플로우 구현', '비개발자 스카우트 즉시 사용 가능 · @st.cache_data 캐싱 · pages/ 모듈 분리 · Streamlit Cloud 배포');

  // URL 배너
  s.addShape(pptx.ShapeType.roundRect, { x:0.35,y:1.18,w:12.6,h:0.48,
    fill:{color:D.dark}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.03 });
  s.addText('🌐  streamlit.app  ›  epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8', {
    x:0.5,y:1.25,w:12.3,h:0.34, fontSize:9.5,bold:true,color:D.mblue,fontFace:F });

  // 페이지 그리드 (강조 페이지)
  const highlight = [1,2,4,5]; // 0-indexed
  const pages = [
    {ic:'🏠',n:'홈',d:'KPI 요약 · 프로젝트 개요'},
    {ic:'👤',n:'선수 즉시 분석',d:'이름 검색 → 스탯·퍼센타일'},
    {ic:'💰',n:'S2 저평가 탐색기',d:'비율 필터·포지션별 조회'},
    {ic:'🔍',n:'선수 유형 탐색기',d:'S3 클러스터 유사선수 검색'},
    {ic:'✈️',n:'이적 인텔리전스',d:'P8 적응 확률·리스크 배너'},
    {ic:'📉',n:'강등권 탐색기',d:'P3 팀별 강등 리스크'},
    {ic:'⚖️',n:'선수 비교',d:'레이더 차트 2-3명 비교'},
    {ic:'🏟️',n:'팀 프로파일',d:'ELO 추이·스쿼드 분석'},
    {ic:'📊',n:'선수 통계 순위',d:'리그 전체 스탯 랭킹'},
    {ic:'📅',n:'시즌 개요',d:'시즌별 트렌드'},
    {ic:'🏆',n:'역대 기록',d:'9시즌 통합 레코드'},
    {ic:'🧠',n:'SHAP 설명',d:'예측 근거·피처 기여도'},
  ];
  const cw=3.1, rh=0.85;
  pages.forEach((p,i) => {
    const col=i%4, row=Math.floor(i/4);
    const x=0.35+col*(cw+0.16), y=1.82+row*(rh+0.08);
    const isHL = highlight.includes(i);
    card(s,x,y,cw,rh,isHL?D.lblue:D.lgray,isHL?D.mblue:D.bord);
    if(isHL) s.addShape(pptx.ShapeType.rect, { x,y,w:0.04,h:rh, fill:{color:D.blue} });
    s.addText(`${p.ic}  ${p.n}`, { x:x+0.1,y:y+0.1,w:cw-0.2,h:0.3, fontSize:9.5,bold:true,color:D.dark,fontFace:F });
    s.addText(p.d, { x:x+0.1,y:y+0.42,w:cw-0.2,h:0.35, fontSize:8.5,color:D.slate,fontFace:F });
  });

  // 기술 하이라이트
  card(s,0.35,5.52,12.6,0.7,D.lgray,D.bord);
  ['@st.cache_data — 대용량 Parquet 캐싱으로 반응 속도 최적화',
   'pages/ 모듈 분리 — 기능별 독립 구조로 유지보수·확장 용이',
   'ML 14종 결과 전부 대시보드 연동 — 스카우트 1개 URL에서 전체 인사이트 접근'].forEach((t,i) => {
    const x = 0.5+i*4.2;
    s.addShape(pptx.ShapeType.ellipse, { x,y:5.7,w:0.1,h:0.1, fill:{color:D.blue} });
    s.addText(t, { x:x+0.17,y:5.65,w:4.0,h:0.5, fontSize:9,color:D.slate,fontFace:F,valign:'middle' });
  });

  ftr(s,10,'Source: Streamlit Cloud | GitHub: robinho0329/EPL-scouting-report-dashboard-project',
    '스카우트 실무자 검증(김태현) — "즉시 업무 적용 가능" 평가 획득. 파란색 강조 = 핵심 스카우팅 페이지');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 11 — System Architecture
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '06. 시스템 아키텍처');
  ttl(s, 'end-to-end 자동화 아키텍처 — 크롤링부터 대시보드까지', 'GitHub Actions 매일 09:15 KST 자동 학습·커밋·Streamlit 상태 체크 완전 자동화');

  // 5단계 레이어 (가로)
  const layers = [
    {t:'DATA SOURCE',  items:['FBref.com','Transfermarkt'], c:'0C2340', w:1.9},
    {t:'CRAWLERS',     items:['fbref_crawler.py','tm_crawler.py'], c:D.navy, w:2.05},
    {t:'PIPELINE',     items:['aggregate.py','preprocess.py','scout_features.py'], c:'1E5F4A', w:2.1},
    {t:'ML MODELS\n14종', items:['p1~p8 예측','s1~s6 스카우트'], c:'5B2C8E', w:2.25},
    {t:'STREAMLIT\n대시보드', items:['15페이지','Cloud 배포'], c:D.dark, w:2.0},
  ];
  let cx=0.35;
  layers.forEach((ly,i) => {
    s.addShape(pptx.ShapeType.roundRect, { x:cx,y:1.18,w:ly.w,h:2.75,
      fill:{color:ly.c}, line:{color:D.bord,width:0}, rectRadius:0.04 });
    s.addText(ly.t, { x:cx,y:1.26,w:ly.w,h:0.48, fontSize:9,bold:true,color:D.white,fontFace:F,align:'center' });
    ly.items.forEach((item,j) => {
      s.addShape(pptx.ShapeType.roundRect, { x:cx+0.08,y:1.82+j*0.88,w:ly.w-0.16,h:0.78,
        fill:{color:'FFFFFF'}, line:{color:D.bord,width:0.4}, rectRadius:0.03 });
      s.addText(item, { x:cx+0.1,y:1.86+j*0.88,w:ly.w-0.2,h:0.68,
        fontSize:9,color:ly.c,bold:true,fontFace:F,align:'center',valign:'middle' });
    });
    if(i<4) {
      s.addShape(pptx.ShapeType.line, { x:cx+ly.w+0.03,y:2.55,w:0.25,h:0, line:{color:D.gray,width:1.5} });
    }
    cx += ly.w+0.33;
  });

  // 데이터 플로우 레이블
  s.addShape(pptx.ShapeType.line, { x:0.35,y:4.1,w:12.6,h:0, line:{color:D.bord,width:0.5} });
  s.addText('데이터 저장 경로', { x:0.35,y:4.15,w:5,h:0.25, fontSize:9,bold:true,color:D.gray,fontFace:F });
  const paths = [
    {p:'data/raw/',c:D.navy},
    {p:'data/processed/',c:'1E5F4A'},
    {p:'data/features/',c:'5B2C8E'},
    {p:'data/scout/',c:D.dark},
    {p:'dashboard/',c:'0C4A6E'},
  ];
  paths.forEach((pt,i) => {
    const x=0.35+i*2.6;
    s.addShape(pptx.ShapeType.roundRect, { x,y:4.42,w:2.45,h:0.5,
      fill:{color:'F8FAFC'}, line:{color:pt.c,width:0.8}, rectRadius:0.03 });
    s.addText(pt.p, { x:x+0.08,y:4.48,w:2.3,h:0.38, fontSize:9,bold:true,color:pt.c,fontFace:F,align:'center',valign:'middle' });
  });

  // GitHub Actions 블록
  s.addShape(pptx.ShapeType.roundRect, { x:0.35,y:5.12,w:12.6,h:1.08,
    fill:{color:'0D1117'}, line:{color:'30363D',width:0.5}, rectRadius:0.04 });
  s.addText('⚙️  GitHub Actions  —  매일 자동 실행', { x:0.5,y:5.18,w:12.3,h:0.28, fontSize:10,bold:true,color:D.mblue,fontFace:F });
  s.addText('cron: 20:50 UTC  →  실제 실행 09:15 KST  (GitHub free-tier 평균 3h24m 지연 역산 적용)  |  timeout: 90분', {
    x:0.5,y:5.48,w:12.3,h:0.22, fontSize:8.5,color:'8B949E',fontFace:'Courier New' });
  s.addText('📥 체크아웃 (actions/checkout@v6)  →  🐍 Python 3.12 (setup-python@v6)  →  📦 ML 의존성 설치  →  🤖 학습 실행  →  📤 결과 커밋  →  🌐 Playwright 상태 체크', {
    x:0.5,y:5.7,w:12.3,h:0.38, fontSize:8.5,color:'6E7681',fontFace:'Courier New' });

  ftr(s,11,'Source: GitHub Actions, Streamlit Cloud | Node.js 24 기반 (Node.js 20 deprecated 해결)',
    '미팅 → 액션아이템 → 다음날 09:15 KST 자동 학습·커밋·배포 완전 자동화 사이클');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 12 — Model Performance Summary
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '07. 모델 성능 총괄');
  ttl(s, '14종 모델 성능 총괄 — 예측 정확도와 실무 활용성', '개발 기간 6개월, 미팅 기반 반복 개선 — 스카우트 실무 검증 3.7/5.0');

  // 성능 요약 테이블
  s.addShape(pptx.ShapeType.rect, { x:0.35,y:1.18,w:12.6,h:0.38, fill:{color:D.dark} });
  ['모델','목적','핵심 지표','결과','활용 상태'].forEach((h,j) => {
    const xs=[0.45,1.55,4.25,7.45,10.35][j], ws=[1.05,2.65,3.15,2.85,2.45][j];
    s.addText(h, { x:xs,y:1.22,w:ws,h:0.3, fontSize:9,bold:true,color:D.white,fontFace:F });
  });

  const rows = [
    {code:'S2',  name:'시장가치 예측',       metric:'Test R²',   val:'0.876',    use:'✅ 저평가 탐색기 배포', good:true},
    {code:'S3',  name:'포지션별 클러스터링', metric:'Silhouette',val:'0.49 (GK)',use:'✅ 유사선수 탐색기 배포', good:true},
    {code:'P3',  name:'강등 예측',           metric:'Test Acc',  val:'97.5%',    use:'✅ 강등권 탐색기 배포', good:true},
    {code:'P8',  name:'이적 적응 예측',      metric:'AUC',       val:'0.735',    use:'✅ 이적 인텔리전스 배포', good:true},
    {code:'P1',  name:'경기 결과 예측',      metric:'F1 Macro',  val:'0.479',    use:'⚡ 개선 진행 중 (목표 0.52)',  good:false},
    {code:'P5',  name:'선수 클러스터링',     metric:'Silhouette',val:'0.21~0.49',use:'✅ S3로 업그레이드', good:true},
    {code:'S1',  name:'선수 랭킹/스코어',    metric:'WAR 스코어',val:'산출 완료',use:'✅ 선수 순위 대시보드', good:true},
    {code:'P4',  name:'선수 성장 예측',      metric:'방향성',    val:'보조 지표', use:'🔄 추가 개선 예정', good:null},
  ];
  rows.forEach((r,i) => {
    const bg = i%2===0 ? 'F8FAFC' : D.white;
    s.addShape(pptx.ShapeType.rect, { x:0.35,y:1.56+i*0.48,w:12.6,h:0.48, fill:{color:bg} });
    s.addShape(pptx.ShapeType.rect, { x:0.35,y:1.56+i*0.48,w:0.38,h:0.48,
      fill:{color: r.good===true ? D.green : r.good===false ? D.amber : D.gray } });
    s.addText(r.code, { x:0.38,y:1.59+i*0.48,w:0.32,h:0.42, fontSize:8,bold:true,color:D.white,fontFace:F,align:'center',valign:'middle' });
    s.addText(r.name,   { x:1.55,y:1.62+i*0.48,w:2.65,h:0.3, fontSize:9.5,color:D.dark,fontFace:F });
    s.addText(r.metric, { x:4.25,y:1.62+i*0.48,w:3.15,h:0.3, fontSize:9,color:D.gray,fontFace:F });
    s.addText(r.val,    { x:7.45,y:1.62+i*0.48,w:2.85,h:0.3, fontSize:9.5,bold:true,color:D.dark,fontFace:F });
    s.addText(r.use,    { x:10.35,y:1.62+i*0.48,w:2.5,h:0.3, fontSize:9,color:D.slate,fontFace:F });
  });
  s.addShape(pptx.ShapeType.rect, { x:0.35,y:5.4,w:12.6,h:0.02, fill:{color:D.bord} });

  // 스카우트 평가
  card(s,0.35,5.48,6.3,0.72,D.lgray,D.bord);
  s.addText('김태현 스카우트 실무 평가', { x:0.5,y:5.55,w:6.0,h:0.26, fontSize:9,color:D.gray,fontFace:F });
  s.addText('현재 3.7 / 5.0', { x:0.5,y:5.78,w:2.5,h:0.3, fontSize:15,bold:true,color:D.gray,fontFace:F });
  s.addText('→  목표 4.0 / 5.0', { x:3.1,y:5.78,w:3.4,h:0.3, fontSize:15,bold:true,color:D.blue,fontFace:F });

  card(s,6.85,5.48,6.1,0.72,D.lgray,D.bord);
  s.addText('"저평가 탐색기·유사선수 검색은 즉시 실무 활용 가능. P1 경기 예측·P8 이적 리스크 정밀도 보강 후 4.0 목표"', {
    x:7.0,y:5.55,w:5.8,h:0.6, fontSize:9,color:D.slate,fontFace:F,italic:true,valign:'middle',wrap:true });

  ftr(s,12,'Source: FBref, Transfermarkt | 14종 모델 전체 Streamlit Cloud 연동',
    '총 6개월 개발 — 데이터 수집부터 ML 모델링, 대시보드 배포까지 end-to-end 구현 완료');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 13 — Learnings & Limitations
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  hdr(s, '08. 배운 점 · 한계');
  ttl(s, '6개월 개발에서 얻은 핵심 인사이트와 개선 방향', '모델 성능보다 실무 활용성 — 데이터 품질이 모델 품질을 결정');

  // 배운 점 3개
  s.addText('핵심 배움', { x:0.35,y:1.18,w:6.3,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });
  const learns = [
    {n:'01',t:'실무 관점 피처 설계',
     d:'김태현 스카우트 페르소나를 통한 반복 검증. 모델 수치보다 "스카우트가 이 결과를 어떻게 쓸 것인가"가 설계의 출발점. S2 v4 필터링이 대표적 사례 — 38세+ / 유스 잠재력 구분.'},
    {n:'02',t:'성능 < 해석 가능성 + 실무 적용성',
     d:'P8 R²=0.127로 4일 연속 실패 → 미팅 결정으로 이진 분류 전환. "숫자가 좋은 모델"보다 "스카우트가 쓸 수 있는 모델"이 프로젝트 목표. SHAP로 예측 근거 투명화.'},
    {n:'03',t:'자동화가 지속 개선의 인프라',
     d:'GitHub Actions로 미팅→액션아이템→자동 학습→커밋 사이클 구현. 새 시즌 데이터 추가만으로 전체 14종 모델 자동 재학습. 개발자 개입 없이 지속 운영 가능한 구조.'},
  ];
  learns.forEach((l,i) => {
    card(s, 0.35, 1.52+i*1.22, 6.3, 1.1, D.lgray, D.bord);
    s.addShape(pptx.ShapeType.rect, { x:0.35,y:1.52+i*1.22,w:0.5,h:1.1, fill:{color:D.dark} });
    s.addText(l.n, { x:0.35,y:1.58+i*1.22,w:0.5,h:0.52, fontSize:16,bold:true,color:D.white,fontFace:F,align:'center' });
    s.addText(l.t, { x:0.95,y:1.58+i*1.22,w:5.6,h:0.3, fontSize:10.5,bold:true,color:D.dark,fontFace:F });
    s.addText(l.d, { x:0.95,y:1.9+i*1.22,w:5.6,h:0.65, fontSize:9,color:D.slate,fontFace:F,wrap:true });
  });

  // 한계 & 개선 방향
  s.addShape(pptx.ShapeType.line, { x:6.85,y:1.15,w:0,h:4.52, line:{color:D.bord,width:0.6} });
  s.addText('한계 & 개선 방향', { x:6.95,y:1.18,w:6.2,h:0.28, fontSize:10,bold:true,color:D.dark,fontFace:F });

  const issues = [
    {type:'GAP', c:D.red,   t:'P1 F1 0.479',            d:'베팅 odds 데이터 연동으로 0.52+ 목표. 무승부 예측이 근본 난제.'},
    {type:'GAP', c:D.red,   t:'P8 AUC 0.735',            d:'포지션별 분리 모델 + 적응 기간 세분화 (3개월/6개월/시즌말)'},
    {type:'GAP', c:D.amber, t:'xG/xA 고급 지표 부재',    d:'StatsBomb 또는 Opta 연동으로 공격 품질 지표 정밀화'},
    {type:'GAP', c:D.amber, t:'수비 빌드업 지표 부재',    d:'Passes under pressure, progressive passes 추가'},
    {type:'NEXT',c:D.blue,  t:'실시간 데이터 갱신',       d:'경기 직후 스탯 자동 업데이트 파이프라인 구현'},
    {type:'NEXT',c:D.blue,  t:'SHAP 개인화 리포트',       d:'선수별 예측 근거 자동 PDF 생성 — 스카우트 리포트 자동화'},
  ];
  issues.forEach((iss,i) => {
    card(s, 6.95, 1.52+i*0.72, 6.2, 0.62, D.lgray, D.bord);
    s.addShape(pptx.ShapeType.roundRect, { x:6.95,y:1.52+i*0.72,w:0.5,h:0.62,
      fill:{color:iss.c}, line:{color:D.bord,width:0}, rectRadius:0.02 });
    s.addText(iss.type, { x:6.95,y:1.55+i*0.72,w:0.5,h:0.55, fontSize:7,bold:true,color:D.white,fontFace:F,align:'center',valign:'middle' });
    s.addText(iss.t, { x:7.52,y:1.56+i*0.72,w:5.55,h:0.28, fontSize:9.5,bold:true,color:D.dark,fontFace:F });
    s.addText(iss.d, { x:7.52,y:1.82+i*0.72,w:5.55,h:0.26, fontSize:8.5,color:D.gray,fontFace:F });
  });

  ftr(s,13,'Source: 개발 기간 2025.11-2026.04',
    '모델 개선 → 스카우트 검증 → 재개선 루프가 프로젝트 품질을 결정한 핵심 방법론');
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 14 — Summary & Impact
// ═══════════════════════════════════════════════════════════════
{
  const s = pptx.addSlide();
  s.background = { color: D.dark };
  s.addShape(pptx.ShapeType.rect, { x:0,y:0,w:0.06,h:7.5, fill:{color:D.blue} });

  s.addText('SUMMARY', { x:0.4,y:0.35,w:12.5,h:0.3, fontSize:10,color:D.mblue,fontFace:F,charSpacing:5 });
  s.addText('데이터 수집부터 제품화까지\nEPL 스카우팅 AI 시스템 구축 완료', { x:0.4,y:0.65,w:9,h:1.3,
    fontSize:30,bold:true,color:D.white,fontFace:'Calibri Light',lineSpacingMultiple:1.1 });
  s.addShape(pptx.ShapeType.line, { x:0.4,y:2.02,w:4,h:0, line:{color:D.blue,width:1.5} });

  // 핵심 성과 6개 (2×3)
  const achs = [
    {n:'14종', l:'ML 모델 구현'},
    {n:'R²\n0.876', l:'S2 시장가치 예측'},
    {n:'AUC\n0.735', l:'P8 이적 적응'},
    {n:'97.5%', l:'P3 강등 예측 Acc'},
    {n:'15P', l:'Streamlit 대시보드'},
    {n:'3.7→4.0', l:'스카우트 평가 목표'},
  ];
  achs.forEach((a,i) => {
    const col=i%3, row=Math.floor(i/3);
    const x=0.4+col*3.15, y=2.22+row*1.5;
    s.addShape(pptx.ShapeType.roundRect, { x,y,w:2.95,h:1.3,
      fill:{color:'121F2E'}, line:{color:'1E3A5F',width:0.5}, rectRadius:0.04 });
    s.addText(a.n, { x,y:y+0.15,w:2.95,h:0.72, fontSize:26,bold:true,color:D.white,fontFace:F,align:'center' });
    s.addText(a.l, { x,y:y+0.88,w:2.95,h:0.3, fontSize:9,color:D.mblue,fontFace:F,align:'center' });
  });

  // 우측 요약
  s.addText('GitHub', { x:9.95,y:2.28,w:1.5,h:0.28, fontSize:9,color:'475569',fontFace:F });
  s.addText('robinho0329/\nEPL-scouting-report-\ndashboard-project', { x:9.95,y:2.55,w:3.2,h:0.72, fontSize:9,color:'94A3B8',fontFace:F });

  s.addText('Streamlit', { x:9.95,y:3.38,w:1.5,h:0.28, fontSize:9,color:'475569',fontFace:F });
  s.addText('epl-scouting-report-\ndashboard-project\n-ffyb8msh6uafxyyg8txsm8\n.streamlit.app', { x:9.95,y:3.65,w:3.2,h:0.88, fontSize:8.5,color:'94A3B8',fontFace:F });

  s.addText('스택', { x:9.95,y:4.65,w:1.5,h:0.28, fontSize:9,color:'475569',fontFace:F });
  s.addText('Python · XGBoost\nLightGBM · Streamlit\nSelenium · Optuna · SHAP', { x:9.95,y:4.92,w:3.2,h:0.62, fontSize:9,color:'94A3B8',fontFace:F });

  // 하단
  s.addShape(pptx.ShapeType.line, { x:0.4,y:5.55,w:12.5,h:0, line:{color:'1E3A5F',width:0.5} });
  s.addText('FBref · Transfermarkt · 2016/17-2024/25 EPL 9시즌 · 개발 기간 2025.11 ~ 2026.04', {
    x:0.4,y:5.62,w:12.5,h:0.28, fontSize:9,color:'475569',fontFace:F });
  s.addText('"스카우트 실무자가 바로 쓸 수 있는 도구를 만드는 것이 이 프로젝트의 핵심이었습니다"', {
    x:0.4,y:5.98,w:12.5,h:0.38, fontSize:11,color:'64748B',fontFace:F,italic:true });

  s.addText('14 / 14', { x:12.5,y:7.17,w:0.7,h:0.2, fontSize:7.5,color:'334155',fontFace:F,align:'right' });
}

// ─── 저장 ───
pptx.writeFile({ fileName: OUT })
  .then(()=> console.log('✅ v2 저장 완료:', OUT))
  .catch(err=>{ console.error('❌ 오류:', err); process.exit(1); });
