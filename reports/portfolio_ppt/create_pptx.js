'use strict';
const PptxGenJS = require('pptxgenjs');

// ─── 색상 상수 ───
const NAVY  = '1B2A4A';
const BLUE  = '4472C4';
const LBLUE = 'BDD7EE';
const GRAY  = '595959';
const WHITE = 'FFFFFF';
const LGRAY = 'D9D9D9';
const FONT  = 'Calibri';
const TOTAL = 14;

// 출력은 이 스크립트와 같은 디렉터리에 만든다 — 절대경로를 박으면 다른 PC에서 못 쓴다
const OUT = require('path').join(__dirname, 'epl_scout_dashboard_portfolio.pptx');

const pptx = new PptxGenJS();
pptx.layout  = 'LAYOUT_WIDE';
pptx.title   = 'EPL Scout Intelligence Dashboard — Portfolio 2026';
pptx.subject = 'EPL Scouting ML System';
pptx.author  = 'EPL Scout Team';

// ─── 헬퍼 ───
function hdr(s, section) {
  s.addText('EPL Scout Intelligence', {
    x:0.25,y:0.08,w:3.8,h:0.28,
    fontSize:9.5,bold:true,color:BLUE,fontFace:FONT
  });
  if(section) s.addText(section,{
    x:4.0,y:0.08,w:5.3,h:0.28,
    fontSize:8.5,color:GRAY,align:'center',fontFace:FONT
  });
  s.addText('2026.04 | Portfolio | Confidential',{
    x:9.6,y:0.08,w:3.5,h:0.28,
    fontSize:8.5,color:GRAY,align:'right',fontFace:FONT
  });
  s.addShape(pptx.ShapeType.line,{
    x:0.25,y:0.4,w:12.83,h:0,
    line:{color:LBLUE,width:0.8}
  });
}

function insight(s, txt) {
  s.addShape(pptx.ShapeType.roundRect,{
    x:0.25,y:6.5,w:12.83,h:0.75,
    fill:{color:LBLUE},line:{color:'9DC3E6',width:0.8},rectRadius:0.04
  });
  s.addText([
    {text:'KEY INSIGHT  ',options:{bold:true,color:NAVY,fontSize:9}},
    {text:txt,options:{color:NAVY,fontSize:9}}
  ],{x:0.4,y:6.52,w:12.5,h:0.72,fontFace:FONT,valign:'middle',wrap:true});
}

function pgn(s,n){
  s.addText(`${n} / ${TOTAL}`,{
    x:12.6,y:7.18,w:0.65,h:0.2,
    fontSize:8,color:GRAY,align:'right',fontFace:FONT
  });
}
function src(s,t='Source: FBref, Transfermarkt'){
  s.addText(t,{x:0.25,y:7.2,w:9,h:0.2,fontSize:7.5,color:GRAY,fontFace:FONT});
}

function rbox(s,x,y,w,h,bg,border){
  s.addShape(pptx.ShapeType.roundRect,{
    x,y,w,h,fill:{color:bg},
    line:{color:border||LBLUE,width:0.8},rectRadius:0.05
  });
}

// ═══════════════════════════════════════════
// SLIDE 1 — Cover
// ═══════════════════════════════════════════
{
  const s = pptx.addSlide();
  s.background = {color:NAVY};

  // 좌측 accent bar
  s.addShape(pptx.ShapeType.rect,{
    x:0,y:0,w:0.15,h:7.5,fill:{color:BLUE}
  });

  s.addText('EPL Scout Intelligence',{
    x:0.45,y:0.85,w:12.5,h:0.45,
    fontSize:14,bold:false,color:LBLUE,fontFace:FONT,charSpacing:3
  });
  s.addText('Dashboard',{
    x:0.45,y:1.3,w:12.5,h:1.3,
    fontSize:58,bold:true,color:WHITE,fontFace:'Calibri Light'
  });
  s.addShape(pptx.ShapeType.line,{
    x:0.45,y:2.7,w:5,h:0,line:{color:BLUE,width:2.5}
  });
  s.addText('FBref + Transfermarkt 기반 EPL 스카우팅 의사결정 지원 시스템',{
    x:0.45,y:2.9,w:12.0,h:0.38,
    fontSize:13,color:LBLUE,fontFace:FONT
  });

  // Info 항목
  const info=[
    ['프로젝트 기간','2025.11 ~ 2026.04'],
    ['데이터 출처','FBref · Transfermarkt'],
    ['ML 모델','14종 (예측 p1~p8, 스카우트 s1~s6)'],
    ['대시보드','Streamlit 15페이지 Cloud 배포'],
  ];
  info.forEach(([k,v],i)=>{
    const col=i%2, row=Math.floor(i/2);
    const x=0.45+col*6.4, y=3.6+row*0.82;
    s.addText(k,{x,y,w:5.8,h:0.22,fontSize:8.5,color:'7090B8',fontFace:FONT});
    s.addText(v,{x,y:y+0.22,w:5.8,h:0.35,fontSize:12,bold:true,color:WHITE,fontFace:FONT});
  });

  s.addText('Python · XGBoost · LightGBM · Streamlit · Selenium · pandas · Optuna · SHAP',{
    x:0.45,y:6.55,w:12.0,h:0.28,fontSize:9,color:'5578A0',fontFace:FONT
  });
  pgn(s,1);
}

// ═══════════════════════════════════════════
// SLIDE 2 — Executive Summary
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'Executive Summary');

  s.addText('데이터 기반 스카우팅 의사결정 시스템 구축',{
    x:0.25,y:0.5,w:12.83,h:0.48,
    fontSize:22,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('FBref·Transfermarkt 크롤링부터 ML 14종, Streamlit 대시보드까지 end-to-end 스카우팅 인텔리전스 구축',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // Pipeline 흐름도
  const steps=[
    {lbl:'📥 데이터\n수집',sub:'FBref\nTransfermarkt\nSelenium'},
    {lbl:'⚙️ 파이프\n라인',sub:'aggregate\npreprocess\nscout_features'},
    {lbl:'🔧 피처\n엔지니어링',sub:'90분 정규화\n포지션 가중치\n60+ 피처'},
    {lbl:'🤖 ML 모델\n14종',sub:'XGBoost\nLightGBM\n앙상블'},
    {lbl:'📊 Streamlit\n대시보드',sub:'15페이지\n스카우트\n워크플로우'},
  ];
  const bw=2.15, stepY=1.42;
  steps.forEach((st,i)=>{
    const x=0.25+i*(bw+0.26);
    s.addShape(pptx.ShapeType.roundRect,{
      x,y:stepY,w:bw,h:1.85,
      fill:{color:i===4?'1B3A6A':'EEF4FB'},
      line:{color:i===4?BLUE:LBLUE,width:1},rectRadius:0.05
    });
    s.addText(st.lbl,{
      x:x+0.05,y:stepY+0.1,w:bw-0.1,h:0.55,
      fontSize:10,bold:true,color:i===4?WHITE:NAVY,fontFace:FONT,align:'center'
    });
    s.addText(st.sub,{
      x:x+0.05,y:stepY+0.65,w:bw-0.1,h:1.1,
      fontSize:8.5,color:i===4?LBLUE:GRAY,fontFace:FONT,align:'center'
    });
    if(i<4){
      s.addShape(pptx.ShapeType.line,{
        x:x+bw+0.02,y:stepY+0.9,w:0.23,h:0,
        line:{color:BLUE,width:1.5}
      });
    }
  });

  // KPI 수치
  const kpis=[
    {n:'14종',l:'ML 모델'},
    {n:'15 P',l:'대시보드 페이지'},
    {n:'9시즌',l:'EPL 데이터'},
    {n:'60+',l:'피처 엔지니어링'},
    {n:'3.7/5',l:'스카우트 실무 평가'},
  ];
  kpis.forEach((k,i)=>{
    const x=0.25+i*2.57;
    rbox(s,x,3.45,2.37,1.12,'F5F8FC',LBLUE);
    s.addText(k.n,{x,y:3.53,w:2.37,h:0.52,fontSize:26,bold:true,color:BLUE,fontFace:FONT,align:'center'});
    s.addText(k.l,{x,y:4.02,w:2.37,h:0.3,fontSize:9,color:GRAY,fontFace:FONT,align:'center'});
  });

  // 모델 구성 표
  rbox(s,0.25,4.72,12.83,1.5,'F8FBFF',LBLUE);
  s.addText('모델 구성 개요',{x:0.4,y:4.79,w:12.5,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  const models=[
    {code:'P1',name:'경기 결과 예측',perf:'F1 0.479'},
    {code:'P3',name:'강등권 예측',perf:'Acc 97.5%'},
    {code:'P5',name:'선수 클러스터링',perf:'K=6 전체'},
    {code:'P8',name:'이적 적응 예측',perf:'AUC 0.735'},
    {code:'S2',name:'시장가치 예측',perf:'R² 0.876'},
    {code:'S3',name:'포지션별 클러스터',perf:'Sil 0.21~0.49'},
  ];
  models.forEach((m,i)=>{
    const x=0.35+i*2.13;
    s.addShape(pptx.ShapeType.rect,{x,y:5.12,w:0.4,h:0.28,fill:{color:NAVY}});
    s.addText(m.code,{x,y:5.14,w:0.4,h:0.24,fontSize:8,bold:true,color:WHITE,fontFace:FONT,align:'center'});
    s.addText(m.name,{x:x+0.44,y:5.14,w:1.5,h:0.24,fontSize:8.5,color:NAVY,fontFace:FONT});
    s.addText(m.perf,{x:x+0.44,y:5.38,w:1.5,h:0.22,fontSize:8,bold:true,color:BLUE,fontFace:FONT});
  });

  insight(s,'스카우트 실무자가 즉시 사용 가능한 end-to-end 시스템 — 저평가 탐색 · 유사 선수 대체 · 이적 리스크 경고 · 강등권 조기 감지의 4가지 스카우팅 질문을 단일 플랫폼에서 해결');
  src(s,'Source: FBref, Transfermarkt, 2016/17-2024/25 EPL 9시즌'); pgn(s,2);
}

// ═══════════════════════════════════════════
// SLIDE 3 — Problem Definition
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'01. 문제 정의');

  s.addText('전통적 스카우팅의 한계: 주관적 평가에서 데이터 기반 의사결정으로',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('스카우트 실무에서 반복되는 4가지 핵심 질문을 데이터로 해결 — 김태현 스카우트 페르소나 기반 검증',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  const probs=[
    {emoji:'💰',q:'Q1. 저평가된 선수를 어떻게 찾나?',
     prob:'시장가치 대비 실제 기여도가 높은 선수를 눈으로만 발굴하는 데는 한계가 있음',
     sol:'→ S2 시장가치 예측 모델로 예측가/실제가 비율 자동 산출 (R²=0.876)'},
    {emoji:'🔄',q:'Q2. 부상/이적 시 대체 선수는?',
     prob:'유사한 플레이 스타일의 선수를 직관적으로만 판단하면 편향이 생김',
     sol:'→ S3 클러스터링으로 포지션별 유사 선수 즉시 탐색 (Silhouette 0.21~0.49)'},
    {emoji:'⚠️',q:'Q3. 이적 후 적응 실패 리스크는?',
     prob:'해외 리그 → EPL 이적 선수의 적응 여부를 계약 전에 예측하기 어려움',
     sol:'→ P8 이진 분류 모델로 적응 확률 스코어 제공 (AUC 0.735)'},
    {emoji:'📉',q:'Q4. 강등 위험팀을 미리 알 수 있나?',
     prob:'시즌 중반 강등권 여부를 직관에만 의존하면 대응이 늦어짐',
     sol:'→ P3 강등 예측 모델로 19라운드 기준 조기 경보 (Acc 97.5%)'},
  ];
  const bw=5.88,bh=2.05;
  probs.forEach((p,i)=>{
    const col=i%2, row=Math.floor(i/2);
    const x=0.25+col*(bw+0.32), y=1.45+row*(bh+0.15);
    rbox(s,x,y,bw,bh,'F0F5FB',LBLUE);
    s.addText(`${p.emoji}  ${p.q}`,{
      x:x+0.15,y:y+0.1,w:bw-0.3,h:0.35,
      fontSize:11,bold:true,color:NAVY,fontFace:FONT
    });
    s.addText(p.prob,{
      x:x+0.15,y:y+0.5,w:bw-0.3,h:0.62,
      fontSize:9.5,color:GRAY,fontFace:FONT,wrap:true
    });
    s.addText(p.sol,{
      x:x+0.15,y:y+1.13,w:bw-0.3,h:0.42,
      fontSize:9.5,bold:true,color:BLUE,fontFace:FONT
    });
  });

  insight(s,'4가지 스카우팅 질문을 모두 데이터로 답변 — "김태현 스카우트" 페르소나가 각 모델 산출물의 실무 적용 가능성을 단계별 검증. 모델 개선 → 스카우트 검증 → 재개선 루프 반복 적용');
  src(s); pgn(s,3);
}

// ═══════════════════════════════════════════
// SLIDE 4 — Data Collection Pipeline
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'02. 데이터 수집');

  s.addText('FBref + Transfermarkt 체크포인트 기반 크롤링 파이프라인',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('Selenium 기반 동적 크롤러, 레이트 리밋 준수 및 재시작 가능한 체크포인트 구조로 대용량 멀티시즌 수집',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // FBref box
  rbox(s,0.25,1.42,6.15,2.8,'EEF4FB','2980B9');
  s.addShape(pptx.ShapeType.rect,{x:0.25,y:1.42,w:6.15,h:0.42,fill:{color:'2C3E50'}});
  s.addText('⚽  FBref',{x:0.35,y:1.48,w:5.95,h:0.3,fontSize:12,bold:true,color:WHITE,fontFace:FONT});
  ['EPL 경기별 스탯 (매치 레벨)','선수별 시즌 스탯 (슈팅·수비·기여도)','팀 ELO 레이팅 데이터 수집',
   '레이트 리밋: 요청 간 6초 준수','수집 범위: 2016/17 ~ 2024/25 시즌'].forEach((t,i)=>{
    s.addText(`▸  ${t}`,{x:0.4,y:1.95+i*0.43,w:5.85,h:0.38,fontSize:9.5,color:NAVY,fontFace:FONT});
  });

  // Transfermarkt box
  rbox(s,6.6,1.42,6.48,2.8,'EEF4FB','E67E22');
  s.addShape(pptx.ShapeType.rect,{x:6.6,y:1.42,w:6.48,h:0.42,fill:{color:'873600'}});
  s.addText('💶  Transfermarkt',{x:6.7,y:1.48,w:6.3,h:0.3,fontSize:12,bold:true,color:WHITE,fontFace:FONT});
  ['선수 시장가치 (시즌별 스냅샷)','이적 기록 (이적료·출발/도착 팀)','선수 신체 정보 (나이·키·주발)',
   '레이트 리밋: 요청 간 5초 준수','국적·포지션·계약 만료일'].forEach((t,i)=>{
    s.addText(`▸  ${t}`,{x:6.75,y:1.95+i*0.43,w:6.2,h:0.38,fontSize:9.5,color:NAVY,fontFace:FONT});
  });

  // 출력 구조
  rbox(s,0.25,4.38,12.83,1.0,'F5F8FC',LBLUE);
  s.addText('📂  data/raw/ 출력 구조',{x:0.4,y:4.45,w:12.5,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  ['matches/ (경기별)','players/ (선수별)','teams/ (팀별)','market_value/ (시장가치)','transfers/ (이적 기록)'].forEach((t,i)=>{
    s.addText(`📄 ${t}`,{x:0.4+i*2.55,y:4.78,w:2.5,h:0.38,fontSize:9,color:GRAY,fontFace:FONT});
  });

  // 체크포인트
  rbox(s,0.25,5.52,12.83,0.72,'EEF4FB',BLUE);
  s.addText('🔄  체크포인트 재시작 기능  — 네트워크 오류·차단 발생 시 마지막 수집 지점부터 자동 재개. SQLite DB로 진행률 관리, 대용량 9시즌 수집 안정화',{
    x:0.4,y:5.59,w:12.5,h:0.59,fontSize:10,color:NAVY,fontFace:FONT,valign:'middle'
  });

  insight(s,'FBref 6초·Transfermarkt 5초 레이트 리밋 준수 + 체크포인트 재시작으로 9시즌 14,000+ rows 안정적 수집. 재시작 가능 구조가 대용량 EPL 데이터 수집의 핵심 인프라');
  src(s,'Source: FBref.com, Transfermarkt.com'); pgn(s,4);
}

// ═══════════════════════════════════════════
// SLIDE 5 — Data Coverage
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'02. 데이터 커버리지');

  s.addText('2016-2025 EPL 9시즌, 선수-시즌 단위 종합 데이터셋 구축',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('시간 기반 train/val/test 분리로 데이터 누수 완전 차단 — 미래 정보를 과거 학습에 절대 사용하지 않음',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // KPI 수치
  const stats=[
    {n:'3,420+',l:'경기 데이터',s:'EPL 매치 레벨',dark:true},
    {n:'14,000+',l:'선수-시즌 Rows',s:'train+val+test',dark:false},
    {n:'60+',l:'피처',s:'90분 정규화 포함',dark:false},
    {n:'9',l:'시즌',s:'2016/17-2024/25',dark:false},
  ];
  stats.forEach((st,i)=>{
    const x=0.25+i*3.22;
    rbox(s,x,1.42,3.0,1.55,st.dark?NAVY:'EEF4FB',st.dark?BLUE:LBLUE);
    s.addText(st.n,{x,y:1.52,w:3.0,h:0.7,fontSize:30,bold:true,color:st.dark?WHITE:BLUE,fontFace:FONT,align:'center'});
    s.addText(st.l,{x,y:2.22,w:3.0,h:0.3,fontSize:10,bold:true,color:st.dark?LBLUE:NAVY,fontFace:FONT,align:'center'});
    s.addText(st.s,{x,y:2.52,w:3.0,h:0.25,fontSize:8.5,color:st.dark?'8899BB':GRAY,fontFace:FONT,align:'center'});
  });

  // Train/Val/Test Split
  rbox(s,0.25,3.15,7.9,2.32,'F5F8FC',LBLUE);
  s.addText('시간 기반 Train / Val / Test 분리',{
    x:0.4,y:3.22,w:7.6,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT
  });
  const splits=[
    {sp:'Train',se:'2016/17 ~ 2020/21',rows:'8,800건',bg:NAVY,tc:WHITE},
    {sp:'Validation',se:'2021/22 ~ 2022/23',rows:'1,210건',bg:BLUE,tc:WHITE},
    {sp:'Test',se:'2023/24 ~ 2024/25',rows:'1,234건',bg:LBLUE,tc:NAVY},
  ];
  splits.forEach((sp,i)=>{
    const x=0.35+i*2.58;
    s.addShape(pptx.ShapeType.rect,{x,y:3.6,w:2.4,h:0.4,fill:{color:sp.bg}});
    s.addText(sp.sp,{x,y:3.63,w:2.4,h:0.34,fontSize:10,bold:true,color:sp.tc,fontFace:FONT,align:'center'});
    s.addText(sp.se,{x,y:4.08,w:2.4,h:0.28,fontSize:9,color:NAVY,fontFace:FONT,align:'center'});
    s.addText(sp.rows,{x,y:4.36,w:2.4,h:0.28,fontSize:9.5,bold:true,color:BLUE,fontFace:FONT,align:'center'});
    if(i<2) s.addText('→',{x:x+2.43,y:3.75,w:0.18,h:0.4,fontSize:13,bold:true,color:GRAY,fontFace:FONT,align:'center',valign:'middle'});
  });

  // 전처리 단계
  rbox(s,8.4,3.15,4.68,2.32,'F5F8FC',LBLUE);
  s.addText('전처리 주요 단계',{x:8.55,y:3.22,w:4.4,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT});
  ['팀명 표준화 (fuzzy matching)','90분 정규화 (per-90 변환)','결측값 포지션별 median 보정',
   'ELO 레이팅 시즌별 산출','롤링 폼 (최근 3·5·10 경기)'].forEach((t,i)=>{
    s.addText(`▸  ${t}`,{x:8.55,y:3.6+i*0.37,w:4.4,h:0.33,fontSize:9.5,color:GRAY,fontFace:FONT});
  });

  insight(s,'시간 순서 기반 데이터 분리로 데이터 누수 완전 차단 — 파이프라인 3단계(aggregate → preprocess → scout_features) 순차 실행, GitHub Actions로 매일 자동 갱신');
  src(s,'Source: FBref.com, Transfermarkt.com | 파이프라인: aggregate → preprocess → scout_features'); pgn(s,5);
}

// ═══════════════════════════════════════════
// SLIDE 6 — Feature Engineering
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'03. 피처 엔지니어링');

  s.addText('90분 기준 정규화 + 누적 경험 피처로 포지션 중립적 평가 체계 구축',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('출전 시간이 다른 선수 간 공정 비교 — 포지션별 다른 가중치로 평가 정밀도 향상',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  const cats=[
    {t:'공격 지표',c:'C0392B',items:['goals_p90, assists_p90','xG_p90, xA_p90','goal_contributions_p90','shots_on_target_p90']},
    {t:'수비 지표',c:'2980B9',items:['tackles_p90, blocks_p90','clearances_p90','interceptions_p90','fouls_p90']},
    {t:'경험/이적 지표',c:'27AE60',items:['epl_experience','transfer_count, transfer_flag','is_cross_league','source_league (5대 리그)']},
    {t:'팀 지표',c:'8E44AD',items:['ELO rating (avg + last)','rolling form 3/5/10','home/away split','team_dependency_score']},
    {t:'시장가치 지표',c:'D35400',items:['log_mv_prev, mv_change_pct','age_premium (젊은 선수)','young_trajectory','value_momentum']},
    {t:'선수 프로파일',c:'16A085',items:['age, age_sq (비선형)','height_cm, foot_code','is_international','versatility_positions, war_norm']},
  ];
  const bw=3.98,bh=1.88;
  cats.forEach((c,i)=>{
    const col=i%3, row=Math.floor(i/3);
    const x=0.25+col*(bw+0.2), y=1.42+row*(bh+0.12);
    rbox(s,x,y,bw,bh,'F8FBFF',LBLUE);
    s.addShape(pptx.ShapeType.rect,{x,y,w:bw,h:0.35,fill:{color:c.c}});
    s.addText(c.t,{x:x+0.1,y:y+0.05,w:bw-0.2,h:0.27,fontSize:10,bold:true,color:WHITE,fontFace:FONT});
    c.items.forEach((item,j)=>{
      s.addText(`▸  ${item}`,{x:x+0.1,y:y+0.42+j*0.34,w:bw-0.2,h:0.3,fontSize:9,color:GRAY,fontFace:FONT});
    });
  });

  insight(s,'과거 시즌 데이터만 사용하는 엄격한 누수 방지 설계 + 90분 정규화로 출전 시간이 적은 선수도 공정 평가 — 포지션별 분리 모델로 FW·MF·DF·GK 특성 반영');
  src(s); pgn(s,6);
}

// ═══════════════════════════════════════════
// SLIDE 7 — Market Value Model (S2)
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'04. 모델: S2 시장가치 예측');

  s.addText('XGBoost 기반 시장가치 예측 — R² 0.876으로 저평가 선수 자동 탐색',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('43개 피처 | 예측가/실제가 비율 ≥1.5× → 저평가 / ≤0.5× → 과대평가 자동 분류 | 스마트 필터 v4 적용',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // 성능 박스
  rbox(s,0.25,1.42,3.9,2.65,NAVY,BLUE);
  s.addText('XGBoost 성능 (Test Set)',{
    x:0.35,y:1.5,w:3.7,h:0.3,fontSize:10,bold:true,color:LBLUE,fontFace:FONT,align:'center'
  });
  [['R²','0.876'],['MAE','3.4M €'],['MAPE','29.8%'],['학습 데이터','8,800건'],['테스트','1,234건']].forEach(([k,v],i)=>{
    s.addText(k,{x:0.35,y:1.88+i*0.37,w:1.7,h:0.3,fontSize:9,color:'9DB8D8',fontFace:FONT});
    s.addText(v,{x:2.05,y:1.88+i*0.37,w:1.9,h:0.3,fontSize:11,bold:true,color:WHITE,fontFace:FONT,align:'right'});
  });

  // 저평가 Top3
  rbox(s,4.35,1.42,4.0,2.65,'EEF8F0','27AE60');
  s.addShape(pptx.ShapeType.rect,{x:4.35,y:1.42,w:4.0,h:0.38,fill:{color:'27AE60'}});
  s.addText('💚  저평가 Top 3 (예측가/실제가)',{x:4.45,y:1.47,w:3.8,h:0.28,fontSize:9.5,bold:true,color:WHITE,fontFace:FONT});
  [['Oliver Arblaster','Sheffield Utd · MF','6.61×'],
   ['Sam Morsy','Ipswich · MF','4.33×'],
   ['Jakub Stolarczyk','Leicester · GK','3.83×']].forEach(([n,t,r],i)=>{
    s.addText(n,{x:4.45,y:1.9+i*0.62,w:2.6,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
    s.addText(t,{x:4.45,y:2.17+i*0.62,w:2.6,h:0.22,fontSize:8.5,color:GRAY,fontFace:FONT});
    s.addText(r,{x:7.1,y:1.9+i*0.62,w:1.1,h:0.35,fontSize:14,bold:true,color:'27AE60',fontFace:FONT,align:'right'});
  });

  // 과대평가 Top3
  rbox(s,8.55,1.42,4.53,2.65,'FEF0EE','E74C3C');
  s.addShape(pptx.ShapeType.rect,{x:8.55,y:1.42,w:4.53,h:0.38,fill:{color:'E74C3C'}});
  s.addText('❌  과대평가 Top 3 (예측가/실제가)',{x:8.65,y:1.47,w:4.3,h:0.28,fontSize:9.5,bold:true,color:WHITE,fontFace:FONT});
  [['Jacob Greaves','Ipswich · DF','0.21×'],
   ['Ibrahim Sangaré',"Nott'm Forest · MF",'0.22×'],
   ['Manuel Ugarte','Man United · MF','0.23×']].forEach(([n,t,r],i)=>{
    s.addText(n,{x:8.65,y:1.9+i*0.62,w:3.0,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
    s.addText(t,{x:8.65,y:2.17+i*0.62,w:3.0,h:0.22,fontSize:8.5,color:GRAY,fontFace:FONT});
    s.addText(r,{x:11.65,y:1.9+i*0.62,w:1.2,h:0.35,fontSize:14,bold:true,color:'E74C3C',fontFace:FONT,align:'right'});
  });

  // v4 필터
  rbox(s,0.25,4.22,12.83,1.35,'F5F8FC',LBLUE);
  s.addText('v4 스마트 필터 설계',{x:0.4,y:4.29,w:12.5,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  ['🟢 저평가 목록: 38세+ 선수 제외 — 나이 감가상각은 진짜 저평가 신호가 아님',
   '🔴 과대평가 목록: 21세 이하 또는 (22세+출전 1,500분 미만) 유스 잠재력 프리미엄 보정 (18명 제외)',
   '⚙️  최소 900분 출전 선수만 스카우팅 대상 — 617명 최종 산출'].forEach((t,i)=>{
    s.addText(t,{x:0.4,y:4.62+i*0.3,w:12.5,h:0.27,fontSize:9.5,color:GRAY,fontFace:FONT});
  });

  insight(s,'예측가/시장가 비율로 저평가 자동 탐색 — 유스 잠재력·나이 감가상각 필터(v4)로 신호 품질 대폭 개선. S2 저평가 탐색기 대시보드에서 포지션·비율 필터와 함께 즉시 조회 가능');
  src(s,'Source: Transfermarkt 시장가치, FBref 스탯 | XGBoost Test R²=0.876, MAE≈3.4M€'); pgn(s,7);
}

// ═══════════════════════════════════════════
// SLIDE 8 — Player Clustering (S3)
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'04. 모델: S3 선수 클러스터링');

  s.addText('포지션별 K-means 클러스터링으로 선수 유형 분류 및 유사 선수 탐색',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('전체 Silhouette 0.115 → 포지션별 분리로 0.21~0.49 대폭 개선 — 포지션 섞임 문제 완전 해결',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // Before → After
  rbox(s,0.25,1.42,4.9,1.52,'FEF0EE','E74C3C');
  s.addText('❌  Before: 전체 통합 클러스터링',{x:0.35,y:1.5,w:4.7,h:0.3,fontSize:10,bold:true,color:'C0392B',fontFace:FONT});
  s.addText('P5: K=6 (전체 선수 통합)\nSilhouette Score: 0.115\n문제: FW·MF·DF·GK 섞임 → 군집 품질 저하',{
    x:0.35,y:1.82,w:4.7,h:1.0,fontSize:9.5,color:GRAY,fontFace:FONT
  });

  s.addText('→',{x:5.3,y:1.88,w:0.55,h:0.65,fontSize:28,bold:true,color:BLUE,fontFace:FONT,align:'center',valign:'middle'});

  rbox(s,5.95,1.42,7.13,1.52,'EEF8F0','27AE60');
  s.addText('✅  After: 포지션별 분리 클러스터링',{x:6.05,y:1.5,w:6.95,h:0.3,fontSize:10,bold:true,color:'1B5E20',fontFace:FONT});
  [['FW','K=7','0.2085'],['MID','K=7','0.2299'],['DEF','K=4','0.3637'],['GK','K=2','0.4873']].forEach(([pos,k,sil],i)=>{
    const x=6.05+i*1.73;
    rbox(s,x,1.82,1.6,0.95,'F0FFF0','81C784');
    s.addText(pos,{x,y:1.86,w:1.6,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT,align:'center'});
    s.addText(k,{x,y:2.15,w:1.6,h:0.25,fontSize:9,color:GRAY,fontFace:FONT,align:'center'});
    s.addText(sil,{x,y:2.37,w:1.6,h:0.3,fontSize:13,bold:true,color:'27AE60',fontFace:FONT,align:'center'});
  });

  // 활용 시나리오
  rbox(s,0.25,3.1,12.83,2.28,'F5F8FC',LBLUE);
  s.addText('스카우팅 활용 시나리오',{x:0.4,y:3.17,w:12.5,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT});
  [
    {icon:'🔍',t:'유사 선수 즉시 탐색',d:'특정 선수 선택 시 같은 클러스터의 스타일 유사 선수 리스트 자동 출력\n대시보드 "선수 유형 탐색기"에서 실시간 조회 가능'},
    {icon:'🔄',t:'부상/이적 대체 시나리오',d:'주전 부상 시 같은 클러스터 내 대체 후보 즉시 탐색\n이적 예산 제약 조건과 함께 필터링 가능'},
    {icon:'📊',t:'포지션별 선수 유형 분류',d:'FW 7유형 (박스 스트라이커·윙·딥라잉 등)\nMID 7유형 (박스투박스·수비형·창의형 등)'},
  ].forEach((c,i)=>{
    const x=0.35+i*4.27;
    s.addText(`${c.icon}  ${c.t}`,{x,y:3.53,w:4.1,h:0.3,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
    s.addText(c.d,{x,y:3.85,w:4.1,h:1.4,fontSize:9,color:GRAY,fontFace:FONT,wrap:true});
  });

  insight(s,'포지션 섞임 문제 해결로 Silhouette 0.115 → 0.21~0.49 대폭 향상 — GK 0.49는 군집이 매우 명확함을 의미. 부상·이적 시 유사 선수 즉시 탐색으로 스카우트 의사결정 속도 향상');
  src(s); pgn(s,8);
}

// ═══════════════════════════════════════════
// SLIDE 9 — Transfer Adaptation (P8)
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'04. 모델: P8 이적 적응 예측');

  s.addText('이진 분류 전환으로 이적 적응 리스크 예측 — AUC 0.735 달성',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('회귀(R²=0.127) 실패 → 이진 분류 전환 | XGBoost + LogReg + RandomForest 소프트 보팅 앙상블',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // Pivot story
  rbox(s,0.25,1.42,3.8,1.48,'FEF0EE','E74C3C');
  s.addText('❌ 회귀 방식 실패',{x:0.35,y:1.5,w:3.6,h:0.3,fontSize:10,bold:true,color:'C0392B',fontFace:FONT});
  s.addText('R² 0.1736 → 0.1572 → 0.1269\n4일 연속 성능 하락\n"연속 수치 예측 자체가 불가"\n→ 미팅 결정: 이진 분류 전환',{
    x:0.35,y:1.82,w:3.6,h:0.96,fontSize:9,color:GRAY,fontFace:FONT
  });

  s.addText('→\n전환',{x:4.15,y:1.75,w:0.6,h:0.65,fontSize:11,bold:true,color:BLUE,fontFace:FONT,align:'center',valign:'middle'});

  rbox(s,4.85,1.42,3.85,1.48,'EEF8F0','27AE60');
  s.addText('✅ 이진 분류 성공',{x:4.95,y:1.5,w:3.65,h:0.3,fontSize:10,bold:true,color:'1B5E20',fontFace:FONT});
  s.addText('타겟: G+A/90 이전 시즌 80% 유지\n최소 출전 5.5×90분 필터\n앙상블: XGB + LogReg + RF\nAUC 0.735 / F1 0.669',{
    x:4.95,y:1.82,w:3.65,h:0.96,fontSize:9,color:GRAY,fontFace:FONT
  });

  // 핵심 지표
  [['AUC','0.735'],['F1','0.669'],['Recall','0.722'],['Accuracy','0.683']].forEach(([k,v],i)=>{
    const x=9.0+i*1.02;
    rbox(s,x,1.42,0.95,1.48,'EEF4FB',LBLUE);
    s.addText(v,{x,y:1.55,w:0.95,h:0.55,fontSize:18,bold:true,color:BLUE,fontFace:FONT,align:'center'});
    s.addText(k,{x,y:2.12,w:0.95,h:0.6,fontSize:8.5,color:NAVY,fontFace:FONT,align:'center'});
  });

  // 리스크 분류
  rbox(s,0.25,3.07,12.83,2.5,'F5F8FC',LBLUE);
  s.addText('적응 리스크 분류 (1,416명 대상)',{x:0.4,y:3.14,w:12.5,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT});
  [
    {l:'⚠️ High Risk',d:'adapt_proba ≤ 0.40\n영입 신중 검토 필요\n이적 후 적응 실패 가능성 높음',n:'514명',c:'E74C3C',bg:'FEF0EE'},
    {l:'➡️ Medium Risk',d:'0.40 < proba < 0.70\n추가 스카우팅 검증 필요\n포지션·리그·이적 경로 고려',n:'649명',c:'F39C12',bg:'FEF9E7'},
    {l:'✅ Low Risk',d:'adapt_proba ≥ 0.70\n적응 성공 가능성 높음\n영입 추천 후보',n:'253명',c:'27AE60',bg:'EEF8F0'},
  ].forEach((r,i)=>{
    const x=0.35+i*4.27;
    rbox(s,x,3.52,4.0,1.9,r.bg,r.c);
    s.addText(r.l,{x:x+0.1,y:3.59,w:3.8,h:0.3,fontSize:10,bold:true,color:r.c,fontFace:FONT});
    s.addText(r.n,{x:x+0.1,y:3.9,w:3.8,h:0.45,fontSize:22,bold:true,color:r.c,fontFace:FONT,align:'center'});
    s.addText(r.d,{x:x+0.1,y:4.35,w:3.8,h:0.97,fontSize:9,color:GRAY,fontFace:FONT});
  });

  insight(s,'포지션 코드(pos_code) + 이전 시즌 G+A 비율 + source_league(La Liga·Serie A 등)가 상위 피처 — 포지션별 공격 기여 유지 여부 + 리그 레벨 차이가 적응 성공의 핵심');
  src(s,'Source: FBref | 학습 1,132건 / 테스트 284건 | 타겟: G+A/90 이전 시즌 80% 유지'); pgn(s,9);
}

// ═══════════════════════════════════════════
// SLIDE 10 — Match Result Prediction (P1)
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'04. 모델: P1 경기 결과 예측');

  s.addText('ELO + H2H + 롤링 폼 기반 경기 결과 예측 — 무승부 Recall 0%→22% 개선',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('88개 피처, XGBoost Optuna 30 trials, class_weight=balanced로 무승부 클래스 감지 대폭 개선',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // Before / After
  rbox(s,0.25,1.42,3.55,2.45,'F8F8F8',LGRAY);
  s.addText('Baseline (v1)',{x:0.35,y:1.5,w:3.35,h:0.3,fontSize:10,bold:true,color:GRAY,fontFace:FONT,align:'center'});
  [['F1 Macro','0.404'],['Accuracy','53.8%'],['무승부 Recall','0%'],['피처 수','49개']].forEach(([k,v],i)=>{
    s.addText(`${k}`,{x:0.35,y:1.87+i*0.42,w:1.8,h:0.35,fontSize:9.5,color:GRAY,fontFace:FONT});
    s.addText(v,{x:2.2,y:1.87+i*0.42,w:1.45,h:0.35,fontSize:10,bold:true,color:GRAY,fontFace:FONT,align:'right'});
  });

  s.addText('+18.7%\nF1 개선',{x:3.9,y:2.1,w:0.8,h:0.8,fontSize:9,bold:true,color:'27AE60',fontFace:FONT,align:'center',valign:'middle'});

  rbox(s,4.8,1.42,3.55,2.45,'EEF8F0','27AE60');
  s.addText('v2 개선판 (현재)',{x:4.9,y:1.5,w:3.35,h:0.3,fontSize:10,bold:true,color:'1B5E20',fontFace:FONT,align:'center'});
  [['F1 Macro','0.479'],['Accuracy','52.3%'],['무승부 Recall','22%'],['피처 수','88개']].forEach(([k,v],i)=>{
    s.addText(`${k}`,{x:4.9,y:1.87+i*0.42,w:1.8,h:0.35,fontSize:9.5,color:GRAY,fontFace:FONT});
    s.addText(v,{x:6.75,y:1.87+i*0.42,w:1.45,h:0.35,fontSize:10.5,bold:true,color:NAVY,fontFace:FONT,align:'right'});
  });

  // 피처 중요도 bar chart
  rbox(s,8.55,1.42,4.53,2.45,'F5F8FC',LBLUE);
  s.addText('Top 피처 (XGBoost Importance)',{x:8.65,y:1.5,w:4.35,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  [['elo_diff',0.109],['elo_ratio',0.071],['elo_diff_abs',0.031],['home_defense_5',0.018],['season_gd_diff',0.016]].forEach(([f,v],i)=>{
    const bw=v/0.109*3.4;
    s.addShape(pptx.ShapeType.rect,{x:8.65,y:1.87+i*0.38,w:bw,h:0.25,fill:{color:i===0?NAVY:LBLUE}});
    s.addText(f,{x:8.65+bw+0.05,y:1.87+i*0.38,w:4.4-bw-0.1,h:0.25,fontSize:8.5,color:GRAY,fontFace:FONT});
  });

  // 한계 및 활용 방향
  rbox(s,0.25,4.0,12.83,1.55,'FFFBF0','F39C12');
  s.addText('📌  EPL 경기 예측의 본질적 한계 및 활용 방향',{x:0.4,y:4.07,w:12.5,h:0.28,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  ['• EPL 경기 결과 예측은 전문 베팅사도 Accuracy 50-55% 수준 — 무승부 예측이 근본적으로 어려운 3클래스 문제',
   '• v2 무승부 Recall 0% → 22%는 핵심 성과 — 삼중 분류 클래스 불균형 문제 class_weight=balanced로 극복',
   '• 활용 방향: 단독 예측보다 팀 강도 평가 보조 지표, ELO diff가 양 팀 수준 차이를 가장 잘 정량화'].forEach((t,i)=>{
    s.addText(t,{x:0.4,y:4.38+i*0.37,w:12.5,h:0.33,fontSize:9.5,color:GRAY,fontFace:FONT});
  });

  insight(s,'ELO diff가 압도적 1위 피처 (0.109) — 팀 간 실력 차이가 결과를 가장 잘 설명함. F1 0.404→0.479 (+18.7%), 무승부 Recall 0%→22% 달성이 v2의 핵심 성과');
  src(s,'Source: FBref | Train 7,890 / Val 760 / Test 730 경기'); pgn(s,10);
}

// ═══════════════════════════════════════════
// SLIDE 11 — Relegation Prediction (P3)
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'04. 모델: P3 강등 예측');

  s.addText('강등권 예측 Acc 97.5% — 시즌 중반 기반 조기 경보 시스템',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('팀 레벨 30개 피처 | Full-season XGBoost Acc 97.5% / F1 92.3% | 19라운드 Mid-season 경보 모드 구현',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // Two mode boxes
  const modes=[
    {title:'📊 Full-Season 모드',desc:'시즌 종료 후 최종 성적 사용\n강등 여부 사후 평가 및 모델 검증',bg:'F0F5FB',bc:LBLUE,
     res:[['XGBoost','97.5%','92.3%','100%'],['RandomForest','97.5%','92.3%','100%'],['LogReg','100%','100%','100%']],
     val:'val XGB: Acc 97.5%, F1 90.9%'},
    {title:'⚠️ Mid-Season 모드 (19R)',desc:'시즌 중반 데이터만 사용\n강등 위험 조기 경보 — 선제 대응 가능',bg:'FFF8E1',bc:'F39C12',
     res:[['XGBoost','100%','100%','100%'],['RandomForest','100%','100%','100%'],['LogReg','100%','100%','100%']],
     val:'val XGB: Acc 90.0%, F1 60.0%'},
  ];
  modes.forEach((m,mi)=>{
    const x=0.25+mi*6.45;
    rbox(s,x,1.42,6.25,3.72,m.bg,m.bc);
    s.addText(m.title,{x:x+0.15,y:1.5,w:5.95,h:0.32,fontSize:11,bold:true,color:NAVY,fontFace:FONT});
    s.addText(m.desc,{x:x+0.15,y:1.84,w:5.95,h:0.55,fontSize:9,color:GRAY,fontFace:FONT});
    // 헤더
    s.addShape(pptx.ShapeType.rect,{x:x+0.1,y:2.45,w:6.05,h:0.3,fill:{color:NAVY}});
    ['모델','Accuracy','F1','AUC-ROC'].forEach((h,j)=>{
      s.addText(h,{x:x+0.1+j*1.5,y:2.48,w:1.45,h:0.24,fontSize:8.5,bold:true,color:WHITE,fontFace:FONT,align:'center'});
    });
    m.res.forEach((r,ri)=>{
      s.addShape(pptx.ShapeType.rect,{x:x+0.1,y:2.75+ri*0.35,w:6.05,h:0.35,fill:{color:ri%2===0?'EEF4FB':WHITE}});
      r.forEach((val,j)=>{
        s.addText(val,{x:x+0.1+j*1.5,y:2.78+ri*0.35,w:1.45,h:0.28,fontSize:9,color:NAVY,fontFace:FONT,align:'center',bold:j>0});
      });
    });
    s.addText(`※ ${m.val}`,{x:x+0.15,y:3.87,w:5.95,h:0.25,fontSize:8.5,color:GRAY,fontFace:FONT,italic:true});
  });

  // 주요 피처
  rbox(s,0.25,5.28,12.83,1.0,'F5F8FC',LBLUE);
  s.addText('주요 피처 (30개)',{x:0.4,y:5.35,w:12.5,h:0.26,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  ['points, ppg, win_rate','goal_diff, goals_for/against','ELO (avg + last_elo)','form_5 (최근 5경기)',
   'squad 시장가치 (avg+total)','avg_epl_experience','promoted flag','shots_on_target'].forEach((f,i)=>{
    const col=i%4, row=Math.floor(i/4);
    s.addText(`▸ ${f}`,{x:0.4+col*3.2,y:5.65+row*0.28,w:3.1,h:0.26,fontSize:9,color:GRAY,fontFace:FONT});
  });

  insight(s,'Full-season XGBoost Acc 97.5%, F1 92.3% — 팀 레벨 포인트·ELO·시장가치가 강등 예측에 충분한 신호. Mid-season(19R) 경보 시스템도 Test Acc 100% 달성');
  src(s,'Source: FBref | Train 2000-2021 / Val 2021-2023 / Test 2023-2025'); pgn(s,11);
}

// ═══════════════════════════════════════════
// SLIDE 12 — Streamlit Dashboard
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'05. Streamlit 대시보드');

  s.addText('15페이지 Streamlit 대시보드 — 스카우트 실무 워크플로우 구현',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('Streamlit Cloud 배포 완료 — 비개발자 스카우트도 즉시 사용 가능한 인터페이스 설계',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // URL
  rbox(s,0.25,1.37,12.83,0.5,NAVY,BLUE);
  s.addText('🌐  https://epl-scouting-report-dashboard-project-ffyb8msh6uafxyyg8txsm8.streamlit.app',{
    x:0.4,y:1.44,w:12.5,h:0.34,fontSize:9.5,bold:true,color:LBLUE,fontFace:FONT
  });

  // 페이지 그리드 (4×3)
  const pages=[
    {icon:'🏠',n:'홈',d:'프로젝트 개요 · KPI 요약'},
    {icon:'👤',n:'선수 즉시 분석',d:'이름 검색 → 통계·퍼센타일'},
    {icon:'💰',n:'S2 저평가 탐색기',d:'비율 필터 · 포지션별 조회'},
    {icon:'🔍',n:'선수 유형 탐색기',d:'S3 클러스터 유사선수 검색'},
    {icon:'✈️',n:'이적 인텔리전스',d:'P8 적응 확률·리스크 배너'},
    {icon:'📉',n:'강등권 탐색기',d:'P3 팀별 강등 리스크 조회'},
    {icon:'⚖️',n:'선수 비교',d:'2-3명 레이더 차트 비교'},
    {icon:'🏟️',n:'팀 프로파일',d:'ELO 추이 · 스쿼드 분석'},
    {icon:'📊',n:'선수 통계 순위',d:'리그 전체 스탯 랭킹'},
    {icon:'📅',n:'시즌 개요',d:'시즌별 트렌드 분석'},
    {icon:'🏆',n:'역대 기록',d:'9시즌 통합 레코드'},
    {icon:'🧠',n:'SHAP 설명',d:'모델 예측 피처 기여도'},
  ];
  const cw=3.12,rh=0.78;
  pages.forEach((p,i)=>{
    const col=i%4, row=Math.floor(i/4);
    const x=0.25+col*(cw+0.15), y=2.0+row*(rh+0.1);
    const highlight=[1,2,4].includes(i);
    rbox(s,x,y,cw,rh,highlight?'EEF4FB':'F8FBFF',LBLUE);
    s.addText(`${p.icon}  ${p.n}`,{x:x+0.08,y:y+0.06,w:cw-0.16,h:0.3,fontSize:9.5,bold:true,color:NAVY,fontFace:FONT});
    s.addText(p.d,{x:x+0.08,y:y+0.37,w:cw-0.16,h:0.34,fontSize:8.5,color:GRAY,fontFace:FONT});
  });

  insight(s,'스카우트 실무자 검증(김태현) — "즉시 업무에 적용 가능"한 인터페이스. @st.cache_data 대용량 캐싱, pages/ 모듈 분리로 기능 확장 용이. ML 14종 결과 모두 대시보드와 연동');
  src(s,'Source: Streamlit Cloud | GitHub: robinho0329/EPL-scouting-report-dashboard-project'); pgn(s,12);
}

// ═══════════════════════════════════════════
// SLIDE 13 — System Architecture
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'06. 시스템 아키텍처');

  s.addText('크롤링 → 파이프라인 → ML → 대시보드 end-to-end 자동화',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('GitHub Actions 매일 09:15 KST 자동 학습 + Playwright Streamlit 앱 상태 체크 완전 자동화',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // Architecture 흐름
  const layers=[
    {l:'DATA\nSOURCE',items:['FBref.com','Transfermarkt'],c:'2C3E50',w:1.85},
    {l:'CRAWLERS',items:['fbref_crawler.py','tm_crawler.py'],c:'2980B9',w:2.05},
    {l:'PIPELINE',items:['aggregate.py','preprocess.py','scout_features.py'],c:'16A085',w:2.1},
    {l:'ML 14종',items:['p1~p8 예측','s1~s6 스카우트'],c:'8E44AD',w:2.2},
    {l:'STREAMLIT\n대시보드',items:['15페이지','Cloud 배포'],c:'1B2A4A',w:1.9},
  ];
  let cx=0.25;
  layers.forEach((ly,i)=>{
    rbox(s,cx,1.42,ly.w,2.85,ly.c,BLUE);
    s.addText(ly.l,{x:cx,y:1.5,w:ly.w,h:0.48,fontSize:9,bold:true,color:WHITE,fontFace:FONT,align:'center'});
    ly.items.forEach((item,j)=>{
      rbox(s,cx+0.08,2.05+j*0.9,ly.w-0.16,0.82,'FFFFFF',LBLUE);
      s.addText(item,{x:cx+0.1,y:2.1+j*0.9,w:ly.w-0.2,h:0.7,fontSize:8.5,color:NAVY,fontFace:FONT,align:'center',valign:'middle'});
    });
    if(i<4){
      s.addShape(pptx.ShapeType.line,{
        x:cx+ly.w+0.03,y:2.78,w:0.27,h:0,
        line:{color:LBLUE,width:1.5}
      });
    }
    cx+=ly.w+0.35;
  });

  // 데이터 흐름
  [['data/raw/','CSV (시즌/팀/선수)','2980B9'],['data/processed/','Parquet (집계 후)','16A085'],
   ['data/features/','Parquet (ML 피처)','8E44AD'],['data/scout/','JSON+Parquet (결과)','1B2A4A']].forEach(([lbl,desc,c],i)=>{
    const x=0.25+i*3.22;
    rbox(s,x,4.42,3.0,0.68,'F5F8FC',c);
    s.addText(lbl,{x:x+0.1,y:4.49,w:2.8,h:0.28,fontSize:9.5,bold:true,color:c,fontFace:FONT});
    s.addText(desc,{x:x+0.1,y:4.74,w:2.8,h:0.28,fontSize:8.5,color:GRAY,fontFace:FONT});
  });

  // GitHub Actions
  rbox(s,0.25,5.25,12.83,1.08,'0D1117','30363D');
  s.addText('🤖  GitHub Actions 자동화',{x:0.4,y:5.32,w:12.5,h:0.28,fontSize:10,bold:true,color:LBLUE,fontFace:FONT});
  ['cron: 매일 20:50 UTC  →  09:15 KST (3h24m 지연 역산 적용)  |  actions/checkout@v6 + setup-python@v6 (Node.js 24)',
   'ML 학습 실행 → git commit/push → Playwright headless Streamlit 상태 체크 → 에러 시 continue-on-error',
   'pandas·scikit-learn·xgboost·lightgbm·optuna·imbalanced-learn·playwright 의존성 자동 설치'].forEach((t,i)=>{
    s.addText(t,{x:0.4,y:5.63+i*0.22,w:12.5,h:0.2,fontSize:8.5,color:'8B949E',fontFace:'Courier New'});
  });

  insight(s,'크롤링→집계→전처리→피처→모델→대시보드의 순차 의존 파이프라인이 GitHub Actions로 완전 자동화 — 미팅 액션아이템 반영 후 다음날 자동 학습·커밋·배포 사이클');
  src(s,'Source: GitHub Actions, Streamlit Cloud | cron: 20:50 UTC = 09:15 KST'); pgn(s,13);
}

// ═══════════════════════════════════════════
// SLIDE 14 — Portfolio Summary
// ═══════════════════════════════════════════
{
  const s=pptx.addSlide();
  hdr(s,'07. 포트폴리오 요약');

  s.addText('데이터 수집부터 제품화까지 — EPL 스카우팅 AI 시스템 구축 완료',{
    x:0.25,y:0.5,w:12.83,h:0.48,fontSize:20,bold:true,color:NAVY,fontFace:FONT
  });
  s.addText('6개월 개발 여정: FBref 크롤링부터 Streamlit Cloud 배포, 스카우트 실무 검증까지',{
    x:0.25,y:0.98,w:12.83,h:0.28,fontSize:10.5,color:GRAY,fontFace:FONT
  });

  // 배운 점
  rbox(s,0.25,1.42,5.95,3.58,'EEF4FB',LBLUE);
  s.addText('✅  배운 점 3가지',{x:0.4,y:1.5,w:5.6,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT});
  [
    {n:'1',t:'실무 관점 피처 설계',d:'김태현 스카우트 페르소나 반복 검증 — 모델이 산출하는 수치를 스카우트가 실제로 활용할 수 있는지 단계별 확인'},
    {n:'2',t:'성능보다 해석 가능성',d:'P8 R²=0.13 실패 → 이진 분류 전환처럼 숫자보다 실무에서 쓸 수 있는 형태 설계가 핵심. SHAP로 예측 근거 투명화'},
    {n:'3',t:'자동화로 지속 개선 구조',d:'GitHub Actions로 미팅→액션아이템→학습→배포 사이클 자동화. 새 시즌 데이터 추가만으로 전체 모델 재학습'},
  ].forEach((l,i)=>{
    s.addShape(pptx.ShapeType.ellipse,{x:0.35,y:1.93+i*1.05,w:0.32,h:0.32,fill:{color:BLUE}});
    s.addText(l.n,{x:0.35,y:1.94+i*1.05,w:0.32,h:0.28,fontSize:10,bold:true,color:WHITE,fontFace:FONT,align:'center'});
    s.addText(l.t,{x:0.75,y:1.93+i*1.05,w:5.3,h:0.3,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
    s.addText(l.d,{x:0.75,y:2.25+i*1.05,w:5.3,h:0.65,fontSize:9,color:GRAY,fontFace:FONT,wrap:true});
  });

  // 한계 및 개선
  rbox(s,6.38,1.42,6.7,3.58,'FFFBF0','F39C12');
  s.addText('🎯  한계와 개선 방향',{x:6.52,y:1.5,w:6.45,h:0.3,fontSize:11,bold:true,color:NAVY,fontFace:FONT});
  [['⚠️','P1 F1 0.479 — 베팅 odds 데이터 추가로 0.52+ 목표'],
   ['⚠️','P8 AUC 0.735 — 포지션별 분리 모델 + 적응 기간 세분화'],
   ['⚠️','xG/xA 고급 지표 부재 — StatsBomb/Opta 데이터 연동 필요'],
   ['📈','실시간 데이터 갱신 — 경기 직후 스탯 자동 업데이트 파이프라인'],
   ['📈','수비수 빌드업 지표 추가 (passes under pressure 등)'],
   ['📈','SHAP 강화 — 개별 선수별 예측 근거 자동 리포트 생성'],
  ].forEach(([icon,t],i)=>{
    s.addText(`${icon}  ${t}`,{x:6.52,y:1.9+i*0.5,w:6.45,h:0.42,fontSize:9.5,color:GRAY,fontFace:FONT});
  });

  // 평가 점수
  rbox(s,0.25,5.12,12.83,1.2,'F0F5FB',BLUE);
  s.addText('스카우트 실무 평가 (김태현)',{x:0.4,y:5.19,w:6.0,h:0.3,fontSize:10,bold:true,color:NAVY,fontFace:FONT});
  s.addText('현재  3.7 / 5.0',{x:0.4,y:5.52,w:3.5,h:0.55,fontSize:20,bold:true,color:GRAY,fontFace:FONT});
  s.addText('→',{x:4.1,y:5.55,w:0.5,h:0.5,fontSize:22,bold:true,color:BLUE,fontFace:FONT,align:'center',valign:'middle'});
  s.addText('목표  4.0 / 5.0',{x:4.75,y:5.52,w:3.5,h:0.55,fontSize:20,bold:true,color:BLUE,fontFace:FONT});
  s.addText('"저평가 탐색기·유사선수 검색은 즉시 실무 활용 가능. P1/P8 정밀도 보강 후 4.0 목표"',{
    x:8.4,y:5.28,w:4.5,h:1.0,fontSize:9,color:GRAY,fontFace:FONT,wrap:true,italic:true
  });

  insight(s,'크롤링·파이프라인·ML·대시보드의 완전한 end-to-end 구현 — 스카우팅 4대 질문(저평가/유사선수/이적적응/강등)을 단일 플랫폼에서 해결한 포트폴리오 프로젝트');
  src(s,'Source: FBref, Transfermarkt | 개발 기간: 2025.11 ~ 2026.04'); pgn(s,14);
}

// ─── 저장 ───
pptx.writeFile({fileName: OUT})
  .then(()=>console.log('✅ 저장 완료:', OUT))
  .catch(err=>{ console.error('❌ 오류:', err); process.exit(1); });
