// Message catalogue for the create-mask stage's panel. Ships with the
// stage; the host merges it at load time. key -> [en-us, zh-cn, zh-tw].

export const strings = {
  'mask.panel':        ['Mask Editor', '蒙版编辑器', '遮罩編輯器'],

  'mask.select':       ['Select a create-mask stage', '选择一个蒙版阶段',
      '選擇一個遮罩階段'],
  'mask.no_stages':    ['No "create-mask" stages in the loaded pipelines. '
      + 'Add one to a pipeline, then Refresh.',
      '已加载的管线中没有“create-mask”阶段。请先添加一个，然后刷新。',
      '已載入的管線中沒有「create-mask」階段。請先新增一個，'
      + '然後重新整理。'],
  'mask.state_live':   ['live', '运行中', '運行中'],
  'mask.waiting_title':['Waiting · {stage}', '等待 · {stage}',
      '等待 · {stage}'],
  'mask.waiting':      ['Waiting for {pipeline} / {stage} to start. It '
      + 'connects automatically when the pipeline runs.',
      '等待 {pipeline} / {stage} 启动。管线运行时将自动连接。',
      '等待 {pipeline} / {stage} 啟動。管線執行時將自動連線。'],
  'mask.editing':      ['Mask · {stage}', '蒙版 · {stage}', '遮罩 · {stage}'],
  'mask.change':       ['Change stage', '切换阶段', '切換階段'],
  'mask.refresh':      ['Refresh', '刷新', '重新整理'],
  'mask.loading':      ['Loading…', '加载中…', '載入中…'],
  'mask.connecting':   ['Connecting…', '连接中…', '連線中…'],
  'mask.reconnecting': ['Reconnecting…', '重新连接中…', '重新連線中…'],
  'mask.no_canvas':    ['Waiting for a reference image or a mask size',
      '等待参考图像或蒙版尺寸', '等待參考影像或遮罩尺寸'],

  // View controls, sharing the preview / compare panels' vocabulary so
  // the same button means the same thing everywhere.
  'mask.zoom_out':     ['Zoom out', '缩小', '縮小'],
  'mask.zoom_in':      ['Zoom in', '放大', '放大'],
  'mask.actual_size':  ['Actual size', '实际大小', '實際大小'],
  'mask.fit':          ['Fit', '适应', '適應'],
  'mask.center':       ['Center', '居中', '置中'],
  'mask.controls_hide':['Hide controls', '隐藏控件', '隱藏控件'],
  'mask.controls_show':['Show controls', '显示控件', '顯示控件'],

  // Painting.
  'mask.hint':         ['Left button paints · right button pans · wheel '
      + 'zooms', '左键绘制 · 右键平移 · 滚轮缩放',
      '左鍵繪製 · 右鍵平移 · 滾輪縮放'],
  'mask.bg_hide':      ['Hide the reference image', '隐藏参考图像',
      '隱藏參考影像'],
  'mask.bg_show':      ['Show the reference image', '显示参考图像',
      '顯示參考影像'],
  'mask.brush_smaller':['Smaller brush', '缩小笔刷', '縮小筆刷'],
  'mask.brush_bigger': ['Larger brush', '放大笔刷', '放大筆刷'],
  'mask.brush_radius': ['Brush radius: {r} px', '笔刷半径：{r} 像素',
      '筆刷半徑：{r} 像素'],
  'mask.softer':       ['Softer edge (lower hardness)', '边缘更柔和（降低硬度）',
      '邊緣更柔和（降低硬度）'],
  'mask.harder':       ['Harder edge (raise hardness)', '边缘更硬（提高硬度）',
      '邊緣更硬（提高硬度）'],
  'mask.hardness':     ['Brush hardness: {h}% — alpha masks only',
      '笔刷硬度：{h}% — 仅用于 alpha 蒙版',
      '筆刷硬度：{h}% — 僅用於 alpha 遮罩'],
  'mask.erase_on':     ['Erasing (click to paint)', '擦除中（点击以绘制）',
      '擦除中（點擊以繪製）'],
  'mask.erase_off':    ['Erase', '擦除', '擦除'],
  'mask.class':        ['Paint class {n}', '绘制类别 {n}', '繪製類別 {n}'],
  'mask.class_bg':     ['Class 0 — background (erases)',
      '类别 0 — 背景（擦除）', '類別 0 — 背景（擦除）'],
  'mask.reset':        ['Reset the mask — discard every edit',
      '重置蒙版 — 放弃所有编辑', '重設遮罩 — 放棄所有編輯'],
  'mask.commit':       ['Commit', '提交', '提交'],
  'mask.commit_tip':   ['Send this mask — the stage emits one beat',
      '发送该蒙版 — 阶段将输出一个节拍',
      '傳送該遮罩 — 階段將輸出一個節拍'],
  'mask.committed':    ['Committed', '已提交', '已提交'],
  'mask.readonly':     ['This stage runs without the editor '
      + '(interactive: false) — the mask is shown but cannot be '
      + 'committed.',
      '该阶段不使用编辑器运行（interactive: false）— 蒙版仅供查看，无法提交。',
      '該階段不使用編輯器執行（interactive: false）— 遮罩僅供檢視，無法提交。'],
};
