// UI string localization for the web client. The catalogue lives here
// (small, inline); the backend keeps its own catalogue for server-
// produced messages (common/i18n.*) and the two share the current
// language via /api/i18n so they stay in sync.
//
// Locale tags are IETF, lowercased with region: en-us / zh-cn / zh-tw.
// Lookups fall back: current locale -> en-us -> the key itself.
//
// The catalogue is keyed by string id; each value is a 3-tuple in locale
// order [en-us, zh-cn, zh-tw]. A "" slot defers a translation (falls back
// to en-us). t(key, params) substitutes {name} placeholders so dynamic
// messages keep correct word order across languages.

const KEY = 'vpipe_lang';

// Supported locales in display order; `label` is the self-name shown in
// the language picker. The first entry is the fallback. The tuple slot
// order in STRINGS matches this order.
export const LOCALES = [
  { tag: 'en-us', label: 'English' },
  { tag: 'zh-cn', label: '简体中文' },
  { tag: 'zh-tw', label: '繁體中文' },
];

// slot index into each STRINGS tuple, by locale tag.
const SLOT = { 'en-us': 0, 'zh-cn': 1, 'zh-tw': 2 };

// key -> [en-us, zh-cn, zh-tw]. Grouped by feature area. Placeholders in
// {braces} are filled by t()'s second argument.
const STRINGS = {
  // ---- Navigation ----
  'nav.pipelines': ['Pipelines', '流水线', '處理管線'],
  'nav.profiler':  ['Profiler', '性能分析', '效能分析'],
  'nav.io':        ['User I/O', '输入输出', '輸入輸出'],
  'nav.database':  ['Database', '数据库', '資料庫'],
  'nav.settings':  ['Settings', '设置', '設定'],

  // ---- Shared verbs / generic labels ----
  'common.save':    ['Save', '保存', '儲存'],
  'common.cancel':  ['Cancel', '取消', '取消'],
  'common.add':     ['Add', '添加', '新增'],
  'common.remove':  ['Remove', '移除', '移除'],
  'common.delete':  ['Delete', '删除', '刪除'],
  'common.create':  ['Create', '创建', '建立'],
  'common.load':    ['Load', '加载', '載入'],
  'common.unload':  ['Unload', '卸载', '卸載'],
  'common.close':   ['Close', '关闭', '關閉'],
  'common.refresh': ['Refresh', '刷新', '重新整理'],
  'common.clear':   ['Clear', '清除', '清除'],
  'common.start':   ['Start', '启动', '啟動'],
  'common.stop':    ['Stop', '停止', '停止'],
  'common.pause':   ['Pause', '暂停', '暫停'],
  'common.open':    ['Open', '打开', '開啟'],
  'common.apply':   ['Apply', '应用', '確定'],
  'common.fit':     ['Fit', '自动缩放', '自動縮放'],
  'common.reset':   ['Reset', '重置', '重設'],
  'common.send':    ['Send', '发送', '傳送'],
  'common.loading': ['Loading…', '加载中…', '載入中…'],
  'common.dismiss': ['Dismiss', '关闭', '關閉'],

  // ---- Settings view ----
  'settings.title':         ['Settings', '设置', '設定'],
  'settings.language':      ['Language', '语言', '語言'],
  'settings.language_desc': ['Interface language. This also sets the '
      + 'language of messages the server sends.',
      '界面语言。同时设置服务器发送消息的语言。',
      '介面語言。同時設定伺服器傳送訊息的語言。'],
  'settings.theme':         ['Color theme', '颜色主题', '色彩主題'],
  'settings.theme_desc':    ['Choose a light or dark appearance, or follow '
      + 'your system ("Auto").',
      '选择浅色或深色外观，或跟随系统（“自动”）。',
      '選擇淺色或深色外觀，或跟隨系統（「自動」）。'],
  'settings.theme_auto':    ['Auto', '自动', '自動'],
  'settings.theme_light':   ['Light', '浅色', '淺色'],
  'settings.theme_dark':    ['Dark', '深色', '深色'],
  'settings.console_history': ['User I/O console history',
      '用户输入输出控制台历史', '使用者輸入輸出主控台歷史'],
  'settings.console_history_desc': ['Maximum number of lines the User I/O '
      + 'console retains. Older lines are dropped first, like a terminal '
      + 'scrollback buffer. Default is 8192.',
      '用户输入输出控制台保留的最大行数。较旧的行会被优先丢弃，'
      + '就像终端的回滚缓冲区。默认值为 8192 行。',
      '使用者輸入輸出主控台保留的最大行數。較舊的行會被優先丟棄，'
      + '就像終端機的回捲緩衝區。預設值為 8192 行。'],
  'settings.log_history':   ['Session log history', '日志历史',
      '日誌歷史'],
  'settings.log_history_desc': ['Maximum number of lines the Session Log '
      + 'view retains. Older lines are dropped first (wraparound buffer). '
      + 'Default is 8192.',
      '日志视图保留的最大行数。较旧的行会被优先丢弃（环形缓冲区）。'
      + '默认值为 8192。',
      '日誌檢視保留的最大行數。較舊的行會被優先丟棄（環形緩衝區）。'
      + '預設值為 8192。'],
  'settings.lines':         ['lines', '行', '行'],
  'settings.limit_positive':['limit must be a positive integer',
      '上限必须为正整数', '上限必須為正整數'],
  'settings.console_updated': ['Console limit updated', '控制台上限已更新',
      '主控台上限已更新'],
  'settings.log_updated':   ['Log limit updated', '日志上限已更新',
      '日誌上限已更新'],
  'settings.save_failed':   ['Save failed: {msg}', '保存失败：{msg}',
      '儲存失敗：{msg}'],
  'settings.saved_current': ['saved · current {n} lines',
      '已保存 · 当前 {n} 行', '已儲存 · 目前 {n} 行'],
  'settings.current_range': ['current {n} lines (range {min}–{max})',
      '当前 {n} 行（范围 {min}–{max}）', '目前 {n} 行（範圍 {min}–{max}）'],
  'settings.no_endpoint':   ['server does not expose the limit endpoint',
      '服务器未提供该上限接口', '伺服器未提供該上限端點'],

  // ---- Startup checks dialog ----
  'startup.title':           ['Startup checks', '启动检查', '啟動檢查'],
  'startup.needs_attention': ['Some startup checks need attention (open '
      + 'System Settings > Privacy & Security):',
      '部分启动检查需要处理（请打开“系统设置 > 隐私与安全性”）：',
      '部分啟動檢查需要處理（請開啟「系統設定 > 隱私權與安全性」）：'],
  'startup.all_passed':      ['All startup checks passed.',
      '所有启动检查均已通过。', '所有啟動檢查均已通過。'],
  'startup.ok':              ['ok', '正常', '正常'],
  'startup.warn':            ['warn', '警告', '警告'],

  // ---- Access-key dialog ----
  'auth.title':       ['Access key required', '需要访问密钥', '需要存取金鑰'],
  'auth.body':        ['This server requires an access key for connections '
      + 'from other computers. Find it printed in the server console '
      + '("Remote access key: …").',
      '此服务器要求来自其他计算机的连接提供访问密钥。'
      + '请在服务器控制台中查找（“Remote access key: …”）。',
      '此伺服器要求來自其他電腦的連線提供存取金鑰。'
      + '請在伺服器主控台中查找（「Remote access key: …」）。'],
  'auth.key':         ['Key', '密钥', '金鑰'],
  'auth.key_ph':      ['8-character key', '8 位字符密钥', '8 位字元金鑰'],
  'auth.connect':     ['Connect', '连接', '連線'],

  // ---- App shell ----
  'app.coming_soon':  ['Coming soon.', '即将推出。', '即將推出。'],

  // ---- Pipeline manager ----
  'pl.stages':         ['Stages', '阶段', '階段'],
  'pl.toolbox':        ['Toolbox', '工具箱', '工具箱'],
  'pl.configuration':  ['Configuration', '配置', '配置'],
  'pl.buf':            ['Buf', '缓冲区图示', '緩衝區圖示'],
  'pl.buf_toggle':     ['Toggle the live buffer-fill overlay (per-edge '
      + 'backlog / capacity while the pipeline runs)',
      '切换实时缓冲填充叠加层（流水线运行时显示每条边的积压量 / 容量）',
      '切換即時緩衝填充疊加層（管線執行時顯示每條邊的積壓量 / 容量）'],
  'pl.buf_interval':   ['Buffer-fill poll interval (seconds)',
      '缓冲填充轮询间隔（秒）', '緩衝填充輪詢間隔（秒）'],
  'pl.filter_ph':      ['Filter…', '筛选…', '篩選…'],
  'pl.hide_toolbox':   ['Hide toolbox', '隐藏工具箱', '隱藏工具箱'],
  'pl.show_toolbox':   ['Show toolbox', '显示工具箱', '顯示工具箱'],
  'pl.select_pipeline':['Select a pipeline', '请选择一个流水线', '請選擇一個管線'],
  'pl.empty_drag':     ['Empty pipeline — drag a stage from the toolbox',
      '空流水线 — 从工具箱拖入一个阶段',
      '空管線 — 從工具箱拖入一個階段'],
  'pl.empty':          ['Empty pipeline', '空流水线', '空管線'],
  'pl.select_stopped': ['Select a stopped pipeline to edit',
      '请选择一个已停止的流水线进行编辑',
      '請選擇一個已停止的管線進行編輯'],
  'pl.chip_hint':      ['\n(drag onto the canvas, or double-click)',
      '\n（拖到画布上，或双击）', '\n（拖到畫布上，或雙擊）'],
  'pl.starting':       ['Starting', '启动中', '啟動中'],
  'pl.pausing':        ['Pausing', '暂停中', '暫停中'],
  'pl.stopping':       ['Stopping', '停止中', '停止中'],
  'pl.inflight_hint':  ['this can take a few seconds on big pipelines '
      + 'while the runtime drains',
      '在大型流水线上，运行时排空期间这可能需要几秒钟',
      '在大型管線上，執行階段排空期間這可能需要幾秒鐘'],
  'pl.no_pipelines':   ['No pipelines. Create or load one.',
      '没有流水线。请创建或加载一个。', '沒有管線。請建立或載入一個。'],
  'pl.no_matches':     ['No matches', '没有匹配项', '沒有相符項目'],
  'pl.stages_total':   ['{n} stage{s} in total', '共 {n} 个流水线级',
      '共 {n} 個階段'],
  'pl.chip_ports':     ['{ins} in / {outs} out', '{ins} 入 / {outs} 出',
      '{ins} 入 / {outs} 出'],
  'pl.add_failed':     ['Add failed: {msg}', '添加失败：{msg}',
      '新增失敗：{msg}'],
  'pl.add_stage_title':['Add "{type}" stage', '添加“{type}”级',
      '新增「{type}」階段'],
  'pl.stage_id':       ['Stage id', '流水线级 id', '管線階段 id'],
  'pl.add_stage_help': ['Created with default config. Its input ports are '
      + 'shown per the stage spec — click an output port, then the target '
      + 'input slot to wire it (inputs fill in index order).',
      '以默认配置创建。其输入端口按阶段规格显示 — 点击一个输出端口，'
      + '再点击目标输入槽即可连线（输入按索引顺序填充）。',
      '以預設配置建立。其輸入埠依階段規格顯示 — 點擊一個輸出埠，'
      + '再點擊目標輸入槽即可連線（輸入依索引順序填入）。'],
  'pl.stage_id_required': ['stage id required', '需要流水线级 id', '需要階段 id'],
  'pl.added':          ['Added {id}', '已添加 {id}', '已新增 {id}'],
  'pl.wire_output_first': ['Click an output port first to start a wire',
      '请先点击一个输出端口以开始连线',
      '請先點擊一個輸出埠以開始連線'],
  'pl.wire_in_order':  ['Connect inputs in order — wire input {n} first',
      '请按顺序连接输入 — 先连接输入 {n}',
      '請依順序連接輸入 — 先連接輸入 {n}'],
  'pl.incompatible':   ['Incompatible types: {from} → {to}',
      '类型不兼容：{from} → {to}', '類型不相容：{from} → {to}'],
  'pl.connected':      ['Connected', '已连接', '已連接'],
  'pl.connect_failed': ['Connect failed: {msg}', '连接失败：{msg}',
      '連接失敗：{msg}'],
  'pl.disconnected':   ['Disconnected', '已断开', '已斷開'],
  'pl.disconnect_failed': ['Disconnect failed: {msg}', '断开失败：{msg}',
      '斷開失敗：{msg}'],
  'pl.select_stage_config': ['Select a stage to view its config.',
      '请选择一个阶段以查看其配置。', '請選擇一個階段以檢視其配置。'],
  'pl.config_unavailable': ['Config unavailable: {msg}', '配置不可用：{msg}',
      '配置不可用：{msg}'],
  'pl.config_readonly':['Pipeline is not stopped — config is read-only.',
      '流水线仍在运行 — 配置为只读。', '管線未停止 — 配置為唯讀。'],
  'pl.remove_stage_title': ['Remove this stage from the pipeline',
      '从流水线中移除此阶段', '從管線中移除此階段'],
  'pl.stop_to_edit':   ['Stop the pipeline to edit', '停止流水线后方可编辑',
      '停止管線後方可編輯'],
  'pl.field_default':  ['default: {val}', '默认：{val}', '預設：{val}'],
  'pl.unset':          ['(unset)', '（未设置）', '（未設定）'],
  'pl.btn_unset':      ['⊘ unset', '⊘ 未设置', '⊘ 未設定'],
  'pl.btn_clear':      ['× clear', '× 清除', '× 清除'],
  'pl.omit_field':     ['omit this field from the config',
      '从配置中省略此字段', '從配置中省略此欄位'],
  'pl.clear_omit':     ['clear (omit this field from the config)',
      '清除（从配置中省略此字段）', '清除（從配置中省略此欄位）'],
  'pl.create_title':   ['Create Pipeline', '创建管线', '建立管線'],
  'pl.pipeline_id':    ['Pipeline id', '流水线 id', '管線 id'],
  'pl.id_required':    ['id required', '需要 id', '需要 id'],
  'pl.create_failed':  ['Create failed: {msg}', '创建失败：{msg}',
      '建立失敗：{msg}'],
  'pl.load_title':     ['Load Pipeline', '加载管线', '載入管線'],
  'pl.file_path':      ['File path', '文件路径', '檔案路徑'],
  'pl.path_required':  ['path required', '需要路径', '需要路徑'],
  'pl.load_failed':    ['Load failed: {msg}', '加载失败：{msg}',
      '載入失敗：{msg}'],
  'pl.op_done':        ['{op} {id}', '{op} {id}', '{op} {id}'],
  'pl.state_stopped':  ['stopped', '已停止', '已停止'],
  'pl.state_paused':   ['paused', '已暂停', '已暫停'],
  'pl.state_running':  ['running', '运行中', '執行中'],
  'pl.vpipeline_files':['{n} .vpipeline file{s} in {cwd}',
      '在 {cwd} 中有 {n} 个 .vpipeline 文件',
      '在 {cwd} 中有 {n} 個 .vpipeline 檔案'],
  'pl.no_vpipeline':   ['No .vpipeline files in {cwd} (type any path)',
      '{cwd} 中没有 .vpipeline 文件（可输入任意路径）',
      '{cwd} 中沒有 .vpipeline 檔案（可輸入任意路徑）'],
  'pl.save_title':     ['Save Pipeline "{id}"', '保存管线“{id}”',
      '儲存管線「{id}」'],
  'pl.save_path_label':['File path (blank = remembered path)',
      '文件路径（留空 = 记住的路径）', '檔案路徑（留空 = 記住的路徑）'],
  'pl.save_hint':      ['A path with no extension gets ".vpipeline" '
      + 'appended automatically.',
      '没有扩展名的路径会自动追加“.vpipeline”。',
      '沒有副檔名的路徑會自動附加「.vpipeline」。'],
  'pl.saved':          ['Saved to {path}', '已保存到 {path}', '已儲存至 {path}'],
  'pl.save_failed':    ['Save failed: {msg}', '保存失败：{msg}',
      '儲存失敗：{msg}'],
  'pl.unload_title':   ['Unload Pipeline', '卸载流水线', '卸載管線'],
  'pl.unload_confirm': ['Unload "{id}"? Unsaved edits are lost.',
      '卸载“{id}”？未保存的修改将丢失。',
      '卸載「{id}」？未儲存的修改將遺失。'],
  'pl.unload_failed':  ['Unload failed: {msg}', '卸载失败：{msg}',
      '卸載失敗：{msg}'],
  'pl.remove_stage_modal': ['Remove Stage', '移除阶段', '移除階段'],
  'pl.remove_stage_confirm': ['Remove stage "{id}"?', '移除阶段“{id}”？',
      '移除階段「{id}」？'],
  'pl.removed':        ['Removed {id}', '已移除 {id}', '已移除 {id}'],
  'pl.remove_failed':  ['Remove failed: {msg}', '移除失败：{msg}',
      '移除失敗：{msg}'],
  'pl.list_failed':    ['List failed: {msg}', '获取列表失败：{msg}',
      '取得列表失敗：{msg}'],
  'pl.detail_failed':  ['Load detail failed: {msg}', '加载详情失败：{msg}',
      '載入詳情失敗：{msg}'],
  'pl.stage_types_failed': ['Stage types failed: {msg}',
      '获取阶段类型失败：{msg}', '取得階段類型失敗：{msg}'],
  'pl.op_failed':      ['{op} failed: {msg}', '{op}失败：{msg}',
      '{op}失敗：{msg}'],
  'pl.invalid_number': ['invalid number for "{key}"', '“{key}”的数字无效',
      '「{key}」的數字無效'],
  'pl.bad_config':     ['Bad config: {msg}', '配置错误：{msg}',
      '配置錯誤：{msg}'],
  'pl.config_applied': ['Config applied', '配置已应用', '配置已套用'],
  'pl.apply_failed':   ['Apply failed: {msg}', '应用失败：{msg}',
      '套用失敗：{msg}'],

  // ---- Profiler ----
  'prof.max_events':   ['Max events captured per worker',
      '每个工作线程捕获的最大事件数', '每個工作執行緒擷取的最大事件數'],
  'prof.reset_title':  ['Clear captured events and start fresh',
      '清除已捕获的事件并重新开始', '清除已擷取的事件並重新開始'],
  'prof.events_per_worker': ['events/worker', '事件/线程', '事件/執行緒'],
  'prof.capture_summary': ['Capture summary', '捕获摘要', '擷取摘要'],
  'prof.events':       ['Events', '事件', '事件'],
  'prof.lanes':        ['Lanes', '通道', '通道'],
  'prof.span':         ['Span', '跨度', '跨度'],
  'prof.dropped':      ['Dropped', '已丢弃', '已丟棄'],
  'prof.hint_click':   ['Click a block (begin/end pair) or marker to '
      + 'inspect it.', '点击一个区块（开始/结束对）或标记以查看详情。',
      '點擊一個區塊（開始/結束對）或標記以檢視詳情。'],
  'prof.hint_start':   ['Start a capture while a pipeline runs.',
      '在流水线运行时开始捕获。', '在管線執行時開始擷取。'],
  'prof.stage':        ['Stage', '流水线级', '管線階段'],
  'prof.lane':         ['Lane', '通道', '通道'],
  'prof.open_suffix':  [' → (open)', ' → （进行中）', ' → （進行中）'],
  'prof.begin':        ['Begin', '开始', '開始'],
  'prof.end':          ['End', '结束', '結束'],
  'prof.still_open':   ['still open at capture end', '在捕获结束时仍未结束',
      '在擷取結束時仍未結束'],
  'prof.duration':     ['Duration', '时长', '時長'],
  'prof.value':        ['Value', '值', '值'],
  'prof.throughput':   ['Throughput', '吞吐量', '吞吐量'],
  'prof.event':        ['Event', '事件', '事件'],
  'prof.transient':    ['  (transient)', '  （瞬时）', '  （瞬時）'],
  'prof.t_rel':        ['t (rel)', 't（相对）', 't（相對）'],
  'prof.t_abs':        ['t (abs)', 't（绝对）', 't（絕對）'],
  'prof.gvid':         ['gvid', 'gvid', 'gvid'],
  'prof.lane_corner':  ['lane', '通道', '通道'],
  'prof.capturing_empty': ['Capturing… run a pipeline to generate events',
      '正在捕获… 运行流水线以生成事件',
      '正在擷取… 執行管線以產生事件'],
  'prof.no_events':    ['No events. Launch a pipeline, Start, then Stop.',
      '没有事件。启动流水线，点击“启动”，然后点击“停止”。',
      '沒有事件。啟動管線，點擊「啟動」，然後點擊「停止」。'],
  'prof.capturing':    ['capturing · {n} events', '正在捕获 · {n} 个事件',
      '正在擷取 · {n} 個事件'],
  'prof.idle':         ['idle', '空闲', '閒置'],
  'prof.events_count': ['{n} events', '{n} 个事件', '{n} 個事件'],
  'prof.lane_overflow':['overflow', '溢出', '溢位'],
  'prof.lane_worker':  ['worker {id}', '工作线程 {id}', '工作執行緒 {id}'],
  'prof.ongoing':      [' (ongoing)', '（进行中）', '（進行中）'],
  'prof.resize_title': ['Drag to resize the details panel',
      '拖动以调整详情面板大小', '拖曳以調整詳情面板大小'],
  'prof.data_failed':  ['Profiler data failed: {msg}', '性能数据获取失败：{msg}',
      '效能資料取得失敗：{msg}'],
  'prof.start_failed': ['Start failed: {msg}', '启动失败：{msg}',
      '啟動失敗：{msg}'],
  'prof.stop_failed':  ['Stop failed: {msg}', '停止失败：{msg}',
      '停止失敗：{msg}'],
  'prof.reset_failed': ['Reset failed: {msg}', '重置失败：{msg}',
      '重設失敗：{msg}'],

  // ---- Database browser ----
  'db.databases':      ['Databases', '数据库', '資料庫'],
  'db.drop_this':      ['Drop this database', '删除此数据库', '刪除此資料庫'],
  'db.no_databases':   ['No databases in the session env.',
      '会话环境中没有数据库。', '工作階段環境中沒有資料庫。'],
  'db.select_db':      ['Select a database on the left.',
      '请在左侧选择一个数据库。', '請在左側選擇一個資料庫。'],
  'db.mode_auto':      ['Auto', '自动', '自動'],
  'db.mode_text':      ['Text', '文本', '文字'],
  'db.mode_number':    ['Number', '数字', '數字'],
  'db.mode_time':      ['Time', '时间', '時間'],
  'db.match_exact':    ['Exact', '精确', '精確'],
  'db.match_range':    ['Range', '范围', '範圍'],
  'db.local_time':     ['Local time ({tz})', '本地时间（{tz}）',
      '本地時間（{tz}）'],
  'db.exact_ph':       ['key value (blank = all)', '键值（留空 = 全部）',
      '鍵值（留空 = 全部）'],
  'db.from_ph':        ['from (blank = unbounded)', '起始（留空 = 不限）',
      '起始（留空 = 不限）'],
  'db.to_ph':          ['to (blank = unbounded)', '结束（留空 = 不限）',
      '結束（留空 = 不限）'],
  'db.interpret_as':   ['Interpret key as', '将键解释为', '將鍵解釋為'],
  'db.match':          ['Match', '匹配', '比對'],
  'db.key':            ['Key', '键', '鍵'],
  'db.from':           ['From', '起始', '起始'],
  'db.to':             ['To', '结束', '結束'],
  'db.run_query':      ['Run query', '运行查询', '執行查詢'],
  'db.query_failed':   ['Query failed: {msg}', '查询失败：{msg}',
      '查詢失敗：{msg}'],
  'db.empty':          ['(empty)', '（空）', '（空）'],
  'db.delete_this':    ['Delete this entry', '删除此条目', '刪除此項目'],
  'db.no_matching':    ['No matching keys.', '没有匹配的键。', '沒有相符的鍵。'],
  'db.page_of':        ['Page {n} of {total}{plus}', '第 {n} / {total}{plus} 页',
      '第 {n} / {total}{plus} 頁'],
  'db.page_n':         ['Page {n}', '第 {n} 页', '第 {n} 頁'],
  'db.first':          ['« First', '« 首页', '« 首頁'],
  'db.first_title':    ['First page', '首页', '首頁'],
  'db.prev':           ['‹ Prev', '‹ 上一页', '‹ 上一頁'],
  'db.prev_title':     ['Previous page', '上一页', '上一頁'],
  'db.next':           ['Next ›', '下一页 ›', '下一頁 ›'],
  'db.next_title':     ['Next page', '下一页', '下一頁'],
  'db.last':           ['Last »', '末页 »', '末頁 »'],
  'db.last_title':     ['Last page', '末页', '末頁'],
  'db.total':          ['{shown} / {total}{plus} in total',
      '共 {shown} / {total}{plus}', '共 {shown} / {total}{plus}'],
  'db.truncated':      [' (truncated)', '（已截断）', '（已截斷）'],
  'db.truncated_title':['scan limit reached', '已达到扫描上限',
      '已達到掃描上限'],
  'db.read_failed':    ['Read failed: {msg}', '读取失败：{msg}',
      '讀取失敗：{msg}'],
  'db.key_not_found':  ['Key not found.', '未找到该键。', '找不到該鍵。'],
  'db.enc_json':       ['FlexData → JSON', 'FlexData → JSON', 'FlexData → JSON'],
  'db.enc_binary':     ['binary · od -t x1 (first 128 B)',
      '二进制 · od -t x1（前 128 B）', '二進位 · od -t x1（前 128 B）'],
  'db.list_failed':    ['Database list failed: {msg}', '数据库列表获取失败：{msg}',
      '資料庫列表取得失敗：{msg}'],
  'db.delete_entry':   ['Delete entry', '删除条目', '刪除項目'],
  'db.delete_entry_confirm': ['Permanently delete this key from "{db}"? '
      + 'This cannot be undone.',
      '从“{db}”中永久删除此键？此操作无法撤销。',
      '從「{db}」中永久刪除此鍵？此操作無法復原。'],
  'db.entry_deleted':  ['Entry deleted', '条目已删除', '項目已刪除'],
  'db.delete_failed':  ['Delete failed: {msg}', '删除失败：{msg}',
      '刪除失敗：{msg}'],
  'db.drop_db':        ['Drop database', '删除数据库', '刪除資料庫'],
  'db.drop_db_confirm':['Permanently drop the entire "{name}" database and '
      + 'every entry in it? This cannot be undone.',
      '永久删除整个“{name}”数据库及其中的所有条目？此操作无法撤销。',
      '永久刪除整個「{name}」資料庫及其中的所有項目？此操作無法復原。'],
  'db.db_dropped':     ['Database dropped', '数据库已删除', '資料庫已刪除'],
  'db.drop_failed':    ['Drop failed: {msg}', '删除失败：{msg}',
      '刪除失敗：{msg}'],

  // ---- I/O workspace ----
  'io.split_options':  ['Split / view options', '拆分 / 视图选项',
      '分割 / 檢視選項'],
  'io.hls':            ['HLS Video', 'HLS 视频', 'HLS 影片'],
  'io.session_log':    ['Session Log', '会话日志', '工作階段日誌'],
  'io.new_view':       ['New view', '新建视图', '新增檢視'],
  'io.add_view':       ['Add a view', '添加一个视图', '新增一個檢視'],
  'io.more_soon':      ['More view types coming soon.',
      '更多视图类型即将推出。', '更多檢視類型即將推出。'],
  'io.split_v':        ['Split vertically (left / right)',
      '垂直拆分（左 / 右）', '垂直分割（左 / 右）'],
  'io.split_h':        ['Split horizontally (top / bottom)',
      '水平拆分（上 / 下）', '水平分割（上 / 下）'],
  'io.close_pane':     ['Close pane', '关闭窗格', '關閉窗格'],

  // ---- Session log view ----
  'log.threshold':     ['Capture threshold (affects future messages only)',
      '捕获阈值（仅影响后续消息）', '擷取門檻（僅影響後續訊息）'],
  'log.set_level_failed': ['Set level failed: {msg}', '设置级别失败：{msg}',
      '設定等級失敗：{msg}'],
  'log.level':         ['Level', '级别', '等級'],

  // ---- User I/O view ----
  'userio.waiting':    ['(waiting for input request)', '（等待输入请求）',
      '（等待輸入請求）'],
  'userio.response_ph':['type a response…', '输入回复…', '輸入回覆…'],
  'userio.newline':    [' new line', ' 换行', ' 換行'],
  'userio.send_word':  [' send', ' 发送', ' 傳送'],
  'userio.input_requested': ['Input requested:', '请求输入：', '請求輸入：'],
  'userio.password_ph':['enter password…', '输入密码…', '輸入密碼…'],
  'userio.markdown':   ['Markdown', 'Markdown', 'Markdown'],
  'userio.markdown_title': ['Render console text as simple Markdown '
      + '(bold / italic / underline, headings, lists, code, tables)',
      '将控制台文本渲染为简单 Markdown'
      + '（粗体 / 斜体 / 下划线、标题、列表、代码、表格）',
      '將主控台文字算繪為簡易 Markdown'
      + '（粗體 / 斜體 / 底線、標題、清單、程式碼、表格）'],

  // ---- HLS video view ----
  'hls.select':        ['Select an HLS stream', '选择一个 HLS 流',
      '選擇一個 HLS 串流'],
  'hls.no_streams':    ['No active HLS streams. Launch a pipeline with an '
      + '"hls-broadcast" stage, then Refresh.',
      '没有活动的 HLS 流。请启动一个包含“hls-broadcast”阶段的管线，'
      + '然后刷新。',
      '沒有作用中的 HLS 串流。請啟動一個包含「hls-broadcast」階段的'
      + '管線，然後重新整理。'],
  'hls.list_failed':   ['Failed to list streams: {msg}', '获取流列表失败：{msg}',
      '取得串流列表失敗：{msg}'],
  'hls.playing':       ['HLS · {stage}', 'HLS · {stage}', 'HLS · {stage}'],
  'hls.change_stream': ['Change stream', '切换流', '切換串流'],
  'hls.playback_error':['Playback error', '播放错误', '播放錯誤'],
  'hls.no_hls':        ['This browser cannot play HLS',
      '此浏览器无法播放 HLS', '此瀏覽器無法播放 HLS'],
  'hls.stream_error':  ['Stream error ({detail})', '流错误（{detail}）',
      '串流錯誤（{detail}）'],
  'hls.load_failed':   ['hls.js failed to load', 'hls.js 加载失败',
      'hls.js 載入失敗'],

  // ---- Status bar ----
  'status.ane':          ['ANE', 'ANE', 'ANE'],
  'status.gpu':          ['GPU', 'GPU', 'GPU'],
  'status.gpu_mem':      ['GPU mem', '显存', '顯示記憶體'],
  'status.mem':          ['MEM', '内存', '記憶體'],
  'status.machine_title':['GPU / chip model', 'GPU / 芯片型号', 'GPU / 晶片型號'],

  // ---- Stage spec docs (overlay; see tOr) ----------------------------
  // These translate text authored in the C++ stage specs (config-key /
  // stage / port doc strings) that arrives over the API in English. The
  // en-us slot mirrors the C++ `.doc` (the API fallback), so an untranslated
  // spec still reads correctly. SEED set -- add stages/keys incrementally;
  // anything absent here falls back to the server's English text.

  // Stage categories.
  'cat.text':     ['Text', '文本', '文字'],
  'cat.audio':    ['Audio', '音频', '音訊'],
  'cat.video':    ['Video', '视频', '視訊'],
  'cat.control':  ['Control', '控制', '控制'],
  'cat.database': ['Database', '数据库', '資料庫'],
  'cat.network':  ['Network', '网络', '網路'],
  'cat.generic':  ['Generic', '通用', '通用'],

  // Stage display names + descriptions.
  'stage.realtime-vqa.name': ['Realtime VQA', '实时视觉问答', '即時視覺問答'],
  'stage.realtime-vqa.doc':  ['Real-time scene VQA: accumulates video frames '
      + 'into scenes (gap/idle-tick boundaries), then prefills once and '
      + 'decodes the configured questions per scene (batched), emitting a '
      + 'FlexData answer bundle.',
      '实时场景视觉问答：将视频帧累积为场景（按间隔/空闲计时边界划分），'
      + '然后对每个场景预填充一次并（批量）解码所配置的问题，输出 FlexData '
      + '答案包。',
      '即時場景視覺問答：將視訊幀累積為場景（依間隔/閒置計時邊界劃分），'
      + '然後對每個場景預填充一次並（批次）解碼所配置的問題，輸出 FlexData '
      + '答案包。'],
  'stage.text-chat.name':    ['Chat', '聊天', '聊天'],
  'stage.text-chat.doc':     ['Conversational LM stage: appends each user '
      + 'turn to a persistent K/V chat context, streams the assistant reply '
      + 'to the UI, and emits a FlexData turn record. /clear resets.',
      '对话语言模型阶段：将每个用户回合追加到持久的 K/V 聊天上下文，'
      + '将助手回复流式传输到界面，并发出 FlexData 回合记录。/clear 可重置。',
      '對話語言模型階段：將每個使用者回合附加到持久的 K/V 聊天上下文，'
      + '將助手回覆串流到介面，並發出 FlexData 回合記錄。/clear 可重設。'],
  'stage.visual-qa.name':    ['Visual Q&A', '视觉问答', '視覺問答'],
  'stage.visual-qa.doc':     ['Sink: loads a vision-language model, encodes '
      + 'incoming image/video frames, and asks the configured questions per '
      + 'round, streaming answers to the UI. 0 oports.',
      '汇聚节点：加载视觉语言模型，编码传入的图像/视频帧，并在每轮提出'
      + '所配置的问题，将答案流式传输到界面。无输出端口。',
      '匯聚節點：載入視覺語言模型，編碼傳入的影像/視訊幀，並在每輪提出'
      + '所配置的問題，將答案串流到介面。無輸出埠。'],
  'stage.audio-transcribe.name': ['Transcribe', '转录', '轉錄'],
  'stage.audio-transcribe.doc':  ['Sink: transcribes each incoming PCM clip '
      + 'with a Qwen3-ASR language model (encoder + greedy/sampled decode) '
      + 'and surfaces the transcript via the UI delegate. 0 oports.',
      '汇聚节点：使用 Qwen3-ASR 语言模型（编码器 + 贪心/采样解码）转录'
      + '每段传入的 PCM 音频，并通过界面委托呈现转录文本。无输出端口。',
      '匯聚節點：使用 Qwen3-ASR 語言模型（編碼器 + 貪婪/取樣解碼）轉錄'
      + '每段傳入的 PCM 音訊，並透過介面委派呈現轉錄文字。無輸出埠。'],

  // realtime-vqa attribute help (compute_dtype etc. omitted -> English).
  'cfg.realtime-vqa.hf_dir': ['VLM model: a models-DB key (registered by '
      + 'model-fetch) or an HF-style model dir; a DB key wins over a '
      + 'same-named path',
      'VLM 模型：models 数据库中的键（由 model-fetch 注册）或 HF 风格的'
      + '模型目录；同名时数据库键优先于路径。',
      'VLM 模型：models 資料庫中的鍵（由 model-fetch 註冊）或 HF 風格的'
      + '模型目錄；同名時資料庫鍵優先於路徑。'],
  'cfg.realtime-vqa.models_db': ['LMDB sub-db model-fetch registers into',
      'model-fetch 注册到的 LMDB 子数据库',
      'model-fetch 註冊到的 LMDB 子資料庫'],
  'cfg.realtime-vqa.coreml_vision_path': ['pre-converted CoreML vision tower '
      + 'path', '预转换的 CoreML 视觉塔路径', '預轉換的 CoreML 視覺塔路徑'],
  'cfg.realtime-vqa.language': ['IETF UI/prompt locale for the built-in scene '
      + 'prompts (en-us | zh-cn | zh-tw); empty inherits the session language',
      '内置场景提示词的 IETF 界面/提示语言（en-us | zh-cn | zh-tw）；'
      + '留空则继承会话语言。',
      '內建場景提示詞的 IETF 介面/提示語言（en-us | zh-cn | zh-tw）；'
      + '留空則繼承工作階段語言。'],
  'cfg.realtime-vqa.max_new_tokens': ['per-question generation budget',
      '每个问题的生成预算（token 数）', '每個問題的生成預算（token 數）'],
  'cfg.realtime-vqa.max_frame_gap_ms': ['inter-frame gap (ms) that closes a '
      + 'scene', '结束一个场景的帧间间隔（毫秒）',
      '結束一個場景的幀間間隔（毫秒）'],
  'cfg.realtime-vqa.idle_ticks_to_end': ['idle ticks with no frame that close '
      + 'a scene (min 2)', '无帧的空闲计时周期数，达到后结束场景（最小为 2）',
      '無幀的閒置計時週期數，達到後結束場景（最小為 2）'],
  'cfg.realtime-vqa.max_frames_per_scene': ['safety cap; closes scene early '
      + 'when reached', '安全上限；达到后提前结束场景',
      '安全上限；達到後提前結束場景'],
  'cfg.realtime-vqa.batched_decode': ['batch question branches that share the '
      + 'prefix', '批量解码共享前缀的问题分支', '批次解碼共享前綴的問題分支'],
  'cfg.realtime-vqa.prev_scene_recap': ['carry prior scene description into '
      + 'the next describe, but only across temporally-continuous scenes; '
      + 'false disables it',
      '将上一场景的描述带入下一次描述，但仅限时间上连续的场景；'
      + '设为 false 可禁用。',
      '將上一場景的描述帶入下一次描述，但僅限時間上連續的場景；'
      + '設為 false 可停用。'],
  'cfg.realtime-vqa.disable_thinking': ['override chat-template thinking '
      + 'default', '覆盖聊天模板的思考（thinking）默认设置',
      '覆寫聊天範本的思考（thinking）預設設定'],
  'cfg.realtime-vqa.questions': ['string or array<string> asked per scene',
      '每个场景提出的问题：字符串或字符串数组',
      '每個場景提出的問題：字串或字串陣列'],
  'cfg.realtime-vqa.question_preamble': ['instruction prepended to every '
      + 'per-question turn (answer-format steer); empty disables',
      '添加到每个问题回合前的指令（用于引导答案格式）；留空则禁用。',
      '加在每個問題回合前的指令（用於引導答案格式）；留空則停用。'],
  'cfg.realtime-vqa.sampler_temperature': ['softmax temperature; <= 0 forces '
      + 'argmax', 'softmax 温度；<= 0 时强制取最大值（argmax）',
      'softmax 溫度；<= 0 時強制取最大值（argmax）'],
  'cfg.realtime-vqa.sampler_top_k': ['keep top-k logits; 0 = disabled',
      '保留前 k 个 logits；0 = 禁用', '保留前 k 個 logits；0 = 停用'],

  // text-chat attribute help.
  'cfg.text-chat.hf_dir': ['model: a models-DB key (registered by model-fetch) '
      + 'or an HF-style model dir; a DB key wins over a same-named path',
      '模型：models 数据库中的键（由 model-fetch 注册）或 HF 风格的模型'
      + '目录；同名时数据库键优先于路径。',
      '模型：models 資料庫中的鍵（由 model-fetch 註冊）或 HF 風格的模型'
      + '目錄；同名時資料庫鍵優先於路徑。'],
  'cfg.text-chat.models_db': ['LMDB sub-db model-fetch registers into',
      'model-fetch 注册到的 LMDB 子数据库',
      'model-fetch 註冊到的 LMDB 子資料庫'],
  'cfg.text-chat.page_tokens': ['ContextManager K/V page size',
      'ContextManager 的 K/V 分页大小', 'ContextManager 的 K/V 分頁大小'],
  'cfg.text-chat.max_pages': ['per-LM page pool capacity (>= 1)',
      '每个语言模型的分页池容量（>= 1）', '每個語言模型的分頁池容量（>= 1）'],
  'cfg.text-chat.max_new_tokens': ['per-turn generation budget (>= 1)',
      '每个回合的生成预算（>= 1）', '每個回合的生成預算（>= 1）'],
  'cfg.text-chat.disable_thinking': ['override chat-template thinking default',
      '覆盖聊天模板的思考（thinking）默认设置',
      '覆寫聊天範本的思考（thinking）預設設定'],
  'cfg.text-chat.sampler': ['decode sampler knobs (temperature/top_k/top_p/...)',
      '解码采样器参数（temperature/top_k/top_p/…）',
      '解碼取樣器參數（temperature/top_k/top_p/…）'],

  // Port help (seed: text-chat + realtime-vqa).
  'port.text-chat.user': ["FlexData string: the user's turn text",
      'FlexData 字符串：用户的回合文本', 'FlexData 字串：使用者的回合文字'],
  'port.text-chat.assistant': ['FlexData {text,prefill_ms,decode_ms,ctx_pos} '
      + 'per turn (downstream optional)',
      '每回合的 FlexData {text,prefill_ms,decode_ms,ctx_pos}（下游可选）',
      '每回合的 FlexData {text,prefill_ms,decode_ms,ctx_pos}（下游可選）'],
  'port.realtime-vqa.frames': ['planar u8 RGB TensorBeat [3,H,W]; sideband '
      + 'timestamp_us drives scene boundaries',
      '平面 u8 RGB TensorBeat [3,H,W]；边带 timestamp_us 驱动场景边界',
      '平面 u8 RGB TensorBeat [3,H,W]；邊帶 timestamp_us 驅動場景邊界'],
  'port.realtime-vqa.scene': ['FlexData per closed scene: questions + answers '
      + '+ frame/timestamp metadata',
      '每个已结束场景的 FlexData：问题 + 答案 + 帧/时间戳元数据',
      '每個已結束場景的 FlexData：問題 + 答案 + 幀/時間戳記中繼資料'],
};

const TAGS = LOCALES.map((l) => l.tag);
const listeners = new Set();
let current = readInitial();

function readInitial() {
  try {
    const v = localStorage.getItem(KEY);
    if (TAGS.includes(v)) { return v; }
  } catch (e) { /* storage blocked -- fall through */ }
  // Fall back to the browser's preferred language, else en-us.
  const nav = (navigator.language || 'en-us').toLowerCase();
  if (nav.startsWith('zh')) {
    return /hant|tw|hk|mo/.test(nav) ? 'zh-tw' : 'zh-cn';
  }
  return 'en-us';
}

export function getLocale() { return current; }
export function locales() { return LOCALES; }

// Translate `key` for the current locale (falling back to en-us, then to
// the key itself). `params` (optional) fills {name} placeholders in the
// string, so dynamic messages keep correct word order per language.
export function t(key, params) {
  const row = STRINGS[key];
  let s;
  if (row) {
    const slot = SLOT[current] != null ? SLOT[current] : 0;
    s = (row[slot] != null && row[slot] !== '') ? row[slot] : row[0];
  } else {
    s = key;
  }
  if (params) {
    s = s.replace(/\{(\w+)\}/g, (m, k) =>
      (params[k] != null ? String(params[k]) : m));
  }
  return s;
}

// Localized override for `key` if the catalogue carries one, else
// `fallback` (a server-provided English string). Used for text that
// originates in the C++ stage spec (config-key / stage / port doc
// strings) and arrives over the API already in English: the catalogue
// translates it WHEN a key exists, and otherwise the English text shows
// through unchanged. This keeps the C++ `.doc` the source of truth and
// makes the per-spec translations purely additive. Key conventions:
//   cfg.<stage-type>.<attr-key>   attribute help
//   stage.<stage-type>.doc        stage description
//   stage.<stage-type>.name       stage display name
//   port.<stage-type>.<port>      port help
//   cat.<category>                stage category label
export function tOr(key, fallback) {
  const row = STRINGS[key];
  if (!row) { return fallback; }
  const slot = SLOT[current] != null ? SLOT[current] : 0;
  const s = (row[slot] != null && row[slot] !== '') ? row[slot] : row[0];
  return (s != null && s !== '') ? s : fallback;
}

// Change the active locale; persists, sets <html lang>, and notifies
// listeners (registered via onLocaleChange) so views can re-render.
export function setLocale(tag) {
  const v = TAGS.includes(tag) ? tag : 'en-us';
  if (v === current) { return v; }
  current = v;
  try { localStorage.setItem(KEY, v); } catch (e) { /* ignore */ }
  document.documentElement.setAttribute('lang', v);
  for (const fn of listeners) {
    try { fn(v); } catch (e) { /* one bad listener shouldn't break others */ }
  }
  return v;
}

// Register a callback fired after the locale changes. Returns an
// unsubscribe function.
export function onLocaleChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
