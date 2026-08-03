// Message catalogue for the preview stage's panel.
//
// These strings ship WITH THE STAGE, not with the front-end app: the
// module exports them and the host merges them into its catalogue at
// load time (see apps/web-ui/web/js/stage-views.js). Same shape as the
// host's own table -- key -> [en-us, zh-cn, zh-tw]; a "" slot falls back
// to English.

export const strings = {
  // Panel label, named by the view spec's `label_key` in
  // preview-view-backend.cc.
  'preview.panel':     ['Live Preview', '实时预览', '即時預覽'],

  'preview.select':    ['Select a preview stage', '选择一个预览阶段',
      '選擇一個預覽階段'],
  'preview.no_stages': ['No "preview" stages in the loaded pipelines. Add '
      + 'one to a pipeline, then Refresh.',
      '已加载的管线中没有“preview”阶段。请先添加一个，然后刷新。',
      '已載入的管線中沒有「preview」階段。請先新增一個，然後重新整理。'],
  'preview.state_live':['live', '运行中', '運行中'],
  'preview.waiting_title':['Waiting · {stage}', '等待 · {stage}',
      '等待 · {stage}'],
  'preview.waiting':   ['Waiting for {pipeline} / {stage} to start. It '
      + 'connects automatically when the pipeline runs.',
      '等待 {pipeline} / {stage} 启动。管线运行时将自动连接。',
      '等待 {pipeline} / {stage} 啟動。管線執行時將自動連線。'],
  'preview.playing':   ['Preview · {stage}', '预览 · {stage}',
      '預覽 · {stage}'],
  'preview.change_stream':['Change stream', '切换流', '切換串流'],
  'preview.connecting':['Connecting…', '连接中…', '連線中…'],
  'preview.reconnecting':['Reconnecting…', '重新连接中…', '重新連線中…'],
  'preview.mse_error': ['Video playback error', '视频播放错误',
      '視訊播放錯誤'],
  'preview.no_video':  ['This browser cannot decode live video — still '
      + 'pictures and audio still play.',
      '此浏览器无法解码实时视频 — 静态画面与音频仍可播放。',
      '此瀏覽器無法解碼即時視訊 — 靜態畫面與音訊仍可播放。'],
  'preview.codec_unsupported':['This browser will not play {codec}',
      '此浏览器不支持 {codec}', '此瀏覽器不支援 {codec}'],
  'preview.audio_only':['Audio only', '仅音频', '僅音訊'],
  'preview.zoom_out':  ['Zoom out', '缩小', '縮小'],
  'preview.zoom_in':   ['Zoom in', '放大', '放大'],
  'preview.actual_size':['Actual size', '实际大小', '實際大小'],
  'preview.center':    ['Center', '居中', '置中'],
  'preview.fit':       ['Fit', '适应', '適應'],
  'preview.refresh':   ['Refresh', '刷新', '重新整理'],
  'preview.loading':   ['Loading…', '加载中…', '載入中…'],
};
