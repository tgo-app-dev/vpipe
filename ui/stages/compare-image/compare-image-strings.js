// Message catalogue for the compare-image stage's panel. Ships with the
// stage; the host merges it at load time. key -> [en-us, zh-cn, zh-tw].

export const strings = {
  'compare.panel':     ['Compare Images', '图像对比', '影像比較'],

  'compare.select':    ['Select a compare stage', '选择一个对比阶段',
      '選擇一個比較階段'],
  'compare.no_stages': ['No "compare-image" stages in the loaded '
      + 'pipelines. Add one to a pipeline, then Refresh.',
      '已加载的管线中没有“compare-image”阶段。请先添加一个，然后刷新。',
      '已載入的管線中沒有「compare-image」階段。請先新增一個，'
      + '然後重新整理。'],
  'compare.state_live':['live', '运行中', '運行中'],
  'compare.waiting_title':['Waiting · {stage}', '等待 · {stage}',
      '等待 · {stage}'],
  'compare.waiting':   ['Waiting for {pipeline} / {stage} to start. It '
      + 'connects automatically when the pipeline runs.',
      '等待 {pipeline} / {stage} 启动。管线运行时将自动连接。',
      '等待 {pipeline} / {stage} 啟動。管線執行時將自動連線。'],
  'compare.showing':   ['Compare · {stage}', '对比 · {stage}',
      '比較 · {stage}'],
  'compare.change':    ['Change stage', '切换阶段', '切換階段'],
  'compare.refresh':   ['Refresh', '刷新', '重新整理'],
  'compare.loading':   ['Loading…', '加载中…', '載入中…'],
  'compare.connecting':['Connecting…', '连接中…', '連線中…'],
  'compare.reconnecting':['Reconnecting…', '重新连接中…', '重新連線中…'],
  'compare.no_images': ['No images yet', '暂无图像', '尚無影像'],

  // View controls, shared with the preview panel's vocabulary.
  'compare.zoom_out':  ['Zoom out', '缩小', '縮小'],
  'compare.zoom_in':   ['Zoom in', '放大', '放大'],
  'compare.actual_size':['Actual size', '实际大小', '實際大小'],
  'compare.fit':       ['Fit', '适应', '適應'],
  'compare.center':    ['Center', '居中', '置中'],

  // Swap, and hiding the rest of the controls.
  'compare.swap_tip':  ['Swap A and B', '交换 A 与 B', '交換 A 與 B'],
  'compare.swap_on':   ['Swapped: A and B are exchanged (click to '
      + 'restore)', '已交换：A 与 B 对调（点击恢复）',
      '已交換：A 與 B 對調（點擊還原）'],
  'compare.controls_hide':['Hide the controls', '隐藏控件', '隱藏控制項'],
  'compare.controls_show':['Show the controls', '显示控件', '顯示控制項'],
  'compare.seam_hide': ['Hide the split line (it stays draggable)',
      '隐藏分割线（仍可拖动）', '隱藏分割線（仍可拖曳）'],
  'compare.seam_show': ['Split line hidden — show it again',
      '分割线已隐藏 — 重新显示', '分割線已隱藏 — 重新顯示'],

  // Comparison modes. The short labels are the button faces; the long
  // ones are their tooltips.
  'compare.mode_a':     ['A', 'A', 'A'],
  'compare.mode_a_tip': ['Image A only', '仅图像 A', '僅影像 A'],
  'compare.mode_b':     ['B', 'B', 'B'],
  'compare.mode_b_tip': ['Image B only', '仅图像 B', '僅影像 B'],
  'compare.mode_lr':     ['A|B', 'A|B', 'A|B'],
  'compare.mode_lr_tip': ['Side by side: A left, B right',
      '并排：A 左，B 右', '並排：A 左，B 右'],
  'compare.mode_tb':     ['A/B', 'A/B', 'A/B'],
  'compare.mode_tb_tip': ['Stacked: A top, B bottom',
      '上下：A 上，B 下', '上下：A 上，B 下'],
  'compare.mode_wipe_v':     ['A◧B', 'A◧B', 'A◧B'],
  'compare.mode_wipe_v_tip': ['Vertical wipe: drag the split -- A left, '
      + 'B right', '垂直分割：拖动分割线 —— A 左，B 右',
      '垂直分割：拖曳分割線 —— A 左，B 右'],
  'compare.mode_wipe_h':     ['A⬒B', 'A⬒B', 'A⬒B'],
  'compare.mode_wipe_h_tip': ['Horizontal wipe: drag the split -- A top, '
      + 'B bottom', '水平分割：拖动分割线 —— A 上，B 下',
      '水平分割：拖曳分割線 —— A 上，B 下'],
};
