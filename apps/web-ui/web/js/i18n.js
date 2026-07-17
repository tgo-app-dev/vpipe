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
  'nav.files':     ['Files', '文件', '檔案'],
  'nav.composer':  ['Composer', '自定义', '自訂'],
  'nav.settings':  ['Settings', '设置', '設定'],

  // ---- Shared verbs / generic labels ----
  'common.save':    ['Save', '保存', '儲存'],
  'common.cancel':  ['Cancel', '取消', '取消'],
  'common.add':     ['Add', '添加', '新增'],
  'common.remove':  ['Remove', '移除', '移除'],
  'common.delete':  ['Delete', '删除', '刪除'],
  'common.create':  ['Create', '创建', '建立'],
  'common.overwrite':['Overwrite', '覆盖', '覆蓋'],
  'common.rename':  ['Rename', '重命名', '重新命名'],
  'common.load':    ['Load', '加载', '載入'],
  'common.unload':  ['Unload', '卸载', '卸載'],
  'common.close':   ['Close', '关闭', '關閉'],
  'common.refresh': ['Refresh', '刷新', '重新整理'],
  'common.clear':   ['Clear', '清除', '清除'],
  'common.start':   ['Start', '启动', '啟動'],
  'common.stop':    ['Stop', '停止', '停止'],
  'common.pause':   ['Pause', '暂停', '暫停'],
  'common.play':    ['Play', '播放', '播放'],
  'common.open':    ['Open', '打开', '開啟'],
  'common.apply':   ['Apply', '应用', '確定'],
  'common.fit':     ['Fit', '自动缩放', '自動縮放'],
  'common.reset':   ['Reset', '重置', '重設'],
  'common.send':    ['Send', '发送', '傳送'],
  'common.loading': ['Loading…', '加载中…', '載入中…'],
  'common.dismiss': ['Dismiss', '关闭', '關閉'],

  // ---- File open/save dialog ----
  'fs.open_title':     ['Open file', '打开文件', '開啟檔案'],
  'fs.save_title':     ['Save file', '保存文件', '儲存檔案'],
  'fs.pick_folder':    ['Choose folder', '选择文件夹', '選擇資料夾'],
  'fs.select_folder':  ['Select folder', '选择此文件夹', '選擇此資料夾'],
  'fs.browse':         ['Browse…', '浏览…', '瀏覽…'],
  'fs.up':             ['Up one level', '上一级', '上一層'],
  'fs.refresh':        ['Refresh', '刷新', '重新整理'],
  'fs.filename':       ['Filename', '文件名', '檔名'],
  'fs.all_files':      ['All files', '所有文件', '所有檔案'],
  'fs.sandboxed':      ['Sandbox', '沙盒', '沙箱'],
  'fs.native':         ['Native', '本地', '本機'],
  'fs.empty':          ['Nothing here', '此处为空', '此處為空'],
  'fs.n_selected':     ['{n} selected', '已选 {n} 项', '已選 {n} 項'],
  'fs.name_required':  ['Enter a filename', '请输入文件名', '請輸入檔名'],
  'fs.select_required':['Select a file', '请选择文件', '請選擇檔案'],
  'fs.list_failed':    ['Browse failed: {msg}', '浏览失败：{msg}',
                        '瀏覽失敗：{msg}'],
  'fs.filter_image':   ['Images', '图片', '圖片'],
  'fs.filter_audio':   ['Audio', '音频', '音訊'],
  'fs.filter_video':   ['Video', '视频', '視訊'],
  'fs.filter_text':    ['Text', '文本', '文字'],
  'fs.new_folder':     ['New folder', '新建文件夹', '新增資料夾'],
  'fs.folder_name':    ['Folder name', '文件夹名称', '資料夾名稱'],
  'fs.mkdir_failed':   ['Could not create folder: {msg}',
                        '无法创建文件夹：{msg}', '無法建立資料夾：{msg}'],
  'fs.overwrite_title':['Overwrite file?', '覆盖文件？', '覆蓋檔案？'],
  'fs.overwrite_msg':  ['"{name}" already exists. Overwrite it?',
                        '“{name}” 已存在，是否覆盖？',
                        '「{name}」已存在，是否覆蓋？'],

  // ---- File browser view ----
  'fb.folders':       ['Folders', '文件夹', '資料夾'],
  'fb.files':         ['Files', '文件', '檔案'],
  'fb.preview':       ['Preview', '预览', '預覽'],
  'fb.new_folder':    ['New folder', '新建文件夹', '新增資料夾'],
  'fb.rename':        ['Rename', '重命名', '重新命名'],
  'fb.pick_to_preview':['Select a file to preview', '选择文件以预览',
                        '選擇檔案以預覽'],
  'fb.no_preview':    ['No preview for this file type', '此文件类型无法预览',
                        '此檔案類型無法預覽'],
  'fb.download':      ['Download', '下载', '下載'],
  'fb.new_folder_name':['Folder name', '文件夹名称', '資料夾名稱'],
  'fb.rename_to':     ['New name', '新名称', '新名稱'],
  'fb.folder_created':['Folder created', '文件夹已创建', '資料夾已建立'],
  'fb.renamed':       ['Renamed', '已重命名', '已重新命名'],
  'fb.select_item':   ['Select an item to rename', '请选择要重命名的项目',
                        '請選擇要重新命名的項目'],
  'fb.truncated':     ['Preview truncated — showing the first {n}',
                        '预览已截断 — 仅显示前 {n}',
                        '預覽已截斷 — 僅顯示前 {n}'],
  'fb.preview_failed':['Preview failed: {msg}', '预览失败：{msg}',
                        '預覽失敗：{msg}'],
  'fb.op_failed':     ['Failed: {msg}', '操作失败：{msg}', '操作失敗：{msg}'],
  'fb.empty_folder':  ['Empty folder', '空文件夹', '空資料夾'],

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
  'pl.rebind_gone':    ['Pipeline "{id}" is no longer loaded',
      '流水线 “{id}” 已不再加载', '管線「{id}」已不再載入'],
  'pl.rebind_pick':    ['Pick a loaded pipeline to edit:',
      '选择一个已加载的流水线进行编辑：', '選擇一個已載入的管線進行編輯：'],
  'pl.rebound':        ['Now editing {id}', '正在编辑 {id}', '正在編輯 {id}'],
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
      + 'input slot to wire it (inputs may be wired in any order).',
      '以默认配置创建。其输入端口按阶段规格显示 — 点击一个输出端口，'
      + '再点击目标输入槽即可连线（输入可按任意顺序连接）。',
      '以預設配置建立。其輸入埠依階段規格顯示 — 點擊一個輸出埠，'
      + '再點擊目標輸入槽即可連線（輸入可按任意順序連接）。'],
  'pl.stage_id_required': ['stage id required', '需要流水线级 id', '需要階段 id'],
  'pl.added':          ['Added {id}', '已添加 {id}', '已新增 {id}'],
  'pl.wire_output_first': ['Click an output port first to start a wire',
      '请先点击一个输出端口以开始连线',
      '請先點擊一個輸出埠以開始連線'],
  'pl.auto_arrange':   ['Auto-arrange', '自动排列', '自動排列'],
  'pl.incompatible':   ['Incompatible types: {from} → {to}',
      '类型不兼容：{from} → {to}', '類型不相容：{from} → {to}'],
  'pl.incompatible_tags': ['Incompatible payload tags: {from} → {to}',
      '负载标签不兼容：{from} → {to}', '負載標籤不相容：{from} → {to}'],
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
  // ---- Model browser ----
  'pl.mb_browse':      ['Browse compatible models', '浏览兼容模型',
      '瀏覽相容模型'],
  'pl.mb_title':       ['Compatible models', '兼容模型', '相容模型'],
  'pl.mb_failed':      ['Failed to load models: {msg}', '加载模型失败：{msg}',
      '載入模型失敗：{msg}'],
  'pl.mb_empty':       ['No compatible installed models. Fetch one with a '
      + 'model-fetch stage.', '没有兼容的已安装模型。请用 model-fetch 阶段获取。',
      '沒有相容的已安裝模型。請用 model-fetch 階段取得。'],
  'pl.mb_g_models':    ['Models', '模型', '模型'],
  'pl.mb_g_supplements': ['Supplements', '附属模型', '附屬模型'],
  'pl.mb_g_datasets':  ['Datasets', '数据集', '資料集'],
  'pl.mb_attaches':    ['attaches to {p}', '附属于 {p}', '附屬於 {p}'],
  'pl.btn_unset':      ['⊘ unset', '⊘ 未设置', '⊘ 未設定'],
  'pl.btn_clear':      ['× clear', '× 清除', '× 清除'],
  'pl.omit_field':     ['omit this field from the config',
      '从配置中省略此字段', '從配置中省略此欄位'],
  'pl.clear_omit':     ['clear (omit this field from the config)',
      '清除（从配置中省略此字段）', '清除（從配置中省略此欄位）'],
  'pl.create_title':   ['Create Pipeline', '创建管线', '建立管線'],
  'pl.pipeline_id':    ['Pipeline id', '流水线 id', '管線 id'],
  'pl.id_required':    ['id required', '需要 id', '需要 id'],
  'pl.new_id':         ['New id', '新 id', '新 id'],
  'pl.rename_stage':   ['Rename stage', '重命名阶段', '重新命名階段'],
  'pl.rename_stage_title': ['Rename Stage', '重命名阶段', '重新命名階段'],
  'pl.stage_renamed':  ['Renamed "{from}" to "{to}"',
      '已将“{from}”重命名为“{to}”', '已將「{from}」重新命名為「{to}」'],
  'pl.rename_pl_title': ['Rename Pipeline', '重命名管线', '重新命名管線'],
  'pl.rename_pl_stopped': ['Stop the pipeline before renaming',
      '重命名前请先停止管线', '重新命名前請先停止管線'],
  'pl.pl_renamed':     ['Renamed pipeline "{from}" to "{to}"',
      '已将管线“{from}”重命名为“{to}”',
      '已將管線「{from}」重新命名為「{to}」'],
  'pl.rename_failed':  ['Rename failed: {msg}', '重命名失败：{msg}',
      '重新命名失敗：{msg}'],
  'pl.duplicate_stage': ['Duplicate', '创建副本', '建立副本'],
  'pl.stage_duplicated': ['Duplicated "{from}" as "{to}"',
      '已将“{from}”复制为“{to}”', '已將「{from}」複製為「{to}」'],
  'pl.duplicate_failed': ['Duplicate failed: {msg}', '创建副本失败：{msg}',
      '建立副本失敗：{msg}'],
  'pl.create_failed':  ['Create failed: {msg}', '创建失败：{msg}',
      '建立失敗：{msg}'],
  'pl.load_title':     ['Load Pipeline', '加载管线', '載入管線'],
  'pl.vpipeline_filter': ['Pipelines (*.vpipeline)', '管线 (*.vpipeline)',
                          '管線 (*.vpipeline)'],
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
  'prof.hide_workers': ['LLM/ANE only', '仅 LLM/ANE', '僅 LLM/ANE'],
  'prof.hide_workers_title': ['Hide worker / overflow lanes; show only the '
      + 'LLM and ANE activity lanes',
      '隐藏工作线程 / 溢出通道，仅显示 LLM 与 ANE 活动通道',
      '隱藏工作執行緒 / 溢位通道，僅顯示 LLM 與 ANE 活動通道'],
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
  // ---- Database value filter (stream) ----
  'db.value_filter':   ['Value filter', '值过滤', '值篩選'],
  'db.add_value_filter': ['+ Add keyword', '+ 添加关键词', '+ 新增關鍵字'],
  'db.vfilter_ph':     ['keyword', '关键词', '關鍵字'],
  'db.cond_includes':  ['includes', '包含', '包含'],
  'db.cond_excludes':  ['excludes', '排除', '排除'],
  'db.cond_regex':     ['matches regex', '匹配正则', '符合正規'],
  'db.cond_regex_not': ['no regex match', '不匹配正则', '不符正規'],
  'db.cond_regex_line': ['line matches regex', '单行匹配正则', '單行符合正規'],
  'db.cond_regex_line_not': ['no line matches regex', '无单行匹配正则',
      '無單行符合正規'],
  'db.highlight':      ['Highlight', '高亮', '標示'],
  'db.highlight_title': ['Highlight the value-filter matches in the value '
      + 'below', '在下方值中高亮值过滤命中的内容',
      '在下方值中標示值篩選命中的內容'],
  'db.match_when':     ['Match', '匹配', '比對'],
  'db.combine_all':    ['all', '全部', '全部'],
  'db.combine_any':    ['any', '任一', '任一'],
  'db.op_and':         ['and', '并且', '並且'],
  'db.op_or':          ['or', '或者', '或者'],
  'db.indent':         ['Indent (group under the row above)',
      '缩进（与上一行分组）', '縮排（與上一行分組）'],
  'db.outdent':        ['Outdent', '取消缩进', '取消縮排'],
  'db.scanning':       ['scanning…', '扫描中…', '掃描中…'],

  // ---- I/O workspace ----
  'io.split_options':  ['Split / view options', '拆分 / 视图选项',
      '分割 / 檢視選項'],
  'io.hls':            ['HLS Video', 'HLS 视频', 'HLS 影片'],
  'io.preview':        ['Live Preview', '实时预览', '即時預覽'],
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

  // ---- Composer view ----
  'composer.pipeline_editor': ['Pipeline editor', '流水线编辑器',
                               '管線編輯器'],
  'composer.pick_pipeline': ['Pipeline', '流水线', '管線'],
  'composer.add':       ['Add panel', '添加面板', '新增面板'],
  'composer.save':      ['Save', '保存', '儲存'],
  'composer.load':      ['Load', '加载', '載入'],
  'composer.save_file': ['Save to file…', '保存为文件…', '儲存為檔案…'],
  'composer.save_pipeline': ['Save with pipeline…', '随流水线保存…',
                             '隨管線儲存…'],
  'composer.load_file': ['Load from file…', '从文件加载…', '從檔案載入…'],
  'composer.load_pipeline': ['Load for pipeline…', '为流水线加载…',
                             '為管線載入…'],
  'composer.clear':     ['Clear all', '清空全部', '清空全部'],
  'composer.empty':     ['Add a panel from the toolbar to compose a '
                         + 'dashboard.', '从工具栏添加面板以组合仪表板。',
                         '從工具列新增面板以組合儀表板。'],
  'composer.float':     ['Float', '浮动', '浮動'],
  'composer.dock_left': ['Dock left', '停靠左侧', '停靠左側'],
  'composer.dock_right':['Dock right', '停靠右侧', '停靠右側'],
  'composer.dock_top':  ['Dock top', '停靠顶部', '停靠頂部'],
  'composer.dock_bottom': ['Dock bottom', '停靠底部', '停靠底部'],
  'composer.maximize':  ['Maximize as background', '最大化为背景',
                         '最大化為背景'],
  'composer.restore':   ['Restore to window', '还原为窗口', '還原為視窗'],
  'composer.close':     ['Close', '关闭', '關閉'],
  'composer.no_pipeline': ['No pipeline available', '没有可用的流水线',
                           '沒有可用的管線'],
  'composer.no_saved':  ['No saved layout for a pipeline', '没有已保存的布局',
                         '沒有已儲存的版面'],
  'composer.saved':     ['Layout saved', '布局已保存', '版面已儲存'],
  'composer.saved_pl':  ['Saved with pipeline "{id}"', '已随流水线“{id}”保存',
                         '已隨管線「{id}」儲存'],
  'composer.loaded':    ['Layout loaded', '布局已加载', '版面已載入'],
  'composer.load_failed': ['Load failed: {msg}', '加载失败：{msg}',
                           '載入失敗：{msg}'],
  'composer.confirm_clear': ['Remove all panels?', '移除所有面板？',
                             '移除所有面板？'],

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
  'userio.media_ph':   ['type a response… (attach, drop or paste image/audio)',
      '输入回复…（可附加、拖入或粘贴图片/音频）',
      '輸入回覆…（可附加、拖入或貼上圖片/音訊）'],
  'userio.image_preview': ['image preview', '图片预览', '圖片預覽'],
  'userio.attach_image': ['Attach image', '附加图片', '附加圖片'],
  'userio.attach_audio': ['Attach audio', '附加音频', '附加音訊'],
  'userio.attach_unsupported': ['Unsupported file type: {name}',
      '不支持的文件类型：{name}', '不支援的檔案類型：{name}'],
  'userio.attach_failed': ['Failed to read file: {name}',
      '读取文件失败：{name}', '讀取檔案失敗：{name}'],
  'userio.attach_too_big': ['File too large: {name} (limit {mb} MB)',
      '文件过大：{name}（上限 {mb} MB）',
      '檔案過大：{name}（上限 {mb} MB）'],
  'userio.markdown':   ['Markdown', 'Markdown', 'Markdown'],
  'userio.thinking':   ['Thinking', '思考过程', '思考過程'],
  'userio.thinking_title': ['Show the model’s reasoning '
      + '("thinking") segments; when off they collapse to a 💭 glyph',
      '显示模型的推理（思考）内容；关闭时折叠为 💭 图标',
      '顯示模型的推理（思考）內容；關閉時摺疊為 💭 圖示'],
  'userio.thinking_hidden': ['thinking hidden — enable the '
      + 'Thinking toggle to view',
      '思考内容已隐藏——打开“思考过程”开关可查看',
      '思考內容已隱藏——開啟「思考過程」開關可檢視'],
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

  // ---- Live Preview view ----
  'preview.select':    ['Select a preview stage', '选择一个预览阶段',
      '選擇一個預覽階段'],
  'preview.no_streams':['No active preview streams. Launch a pipeline with '
      + 'a "preview" stage, then Refresh.',
      '没有活动的预览流。请启动一个包含“preview”阶段的管线，然后刷新。',
      '沒有作用中的預覽串流。請啟動一個包含「preview」階段的管線，'
      + '然後重新整理。'],
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
  'preview.list_failed':['Failed to list streams: {msg}',
      '获取流列表失败：{msg}', '取得串流列表失敗：{msg}'],
  'preview.playing':   ['Preview · {stage}', '预览 · {stage}',
      '預覽 · {stage}'],
  'preview.change_stream':['Change stream', '切换流', '切換串流'],
  'preview.connecting':['Connecting…', '连接中…', '連線中…'],
  'preview.reconnecting':['Reconnecting…', '重新连接中…', '重新連線中…'],
  'preview.mse_error': ['Video playback error', '视频播放错误',
      '視訊播放錯誤'],
  'preview.unsupported':['This browser cannot play the preview (no Media '
      + 'Source Extensions).',
      '此浏览器无法播放预览（不支持 Media Source Extensions）。',
      '此瀏覽器無法播放預覽（不支援 Media Source Extensions）。'],

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
  'cat.visual':   ['Visual', '影像', '影像'],
  'cat.vision':   ['Vision', '视觉', '視覺'],
  'cat.generative': ['Generative', '生成', '生成'],
  'cat.control':  ['Control', '控制', '控制'],
  'cat.database': ['Database', '数据库', '資料庫'],
  'cat.network':  ['Network', '网络', '網路'],
  'cat.preparation': ['Preparation', '准备', '準備'],
  'cat.generic':  ['Generic', '通用', '通用'],

  // Stage display names + descriptions.
  'stage.load-video.name':   ['Load Video', '加载视频', '載入影片'],
  'stage.save-video.name':   ['Save Video', '保存视频', '儲存影片'],
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
  'stage.audio-transcribe.name': ['Transcribe', '语音转录', '語音轉錄'],
  'stage.audio-transcribe.doc':  ['Sink: transcribes each incoming PCM clip '
      + 'with a Qwen3-ASR language model (encoder + greedy/sampled decode) '
      + 'and surfaces the transcript via the UI delegate. 0 oports.',
      '汇聚节点：使用 Qwen3-ASR 语言模型（编码器 + 贪心/采样解码）转录'
      + '每段传入的 PCM 音频，并通过界面委托呈现转录文本。无输出端口。',
      '匯聚節點：使用 Qwen3-ASR 語言模型（編碼器 + 貪婪/取樣解碼）轉錄'
      + '每段傳入的 PCM 音訊，並透過介面委派呈現轉錄文字。無輸出埠。'],
  'stage.load-image.name':    ['Load Image', '加载图片', '載入影像'],
  'stage.save-image.name':    ['Save Image', '保存图片', '儲存影像'],
  'stage.preview.name':       ['Preview', '预览', '預覽'],
  'stage.sampler-select.name': ['Sampler', '采样器', '取樣器'],
  'stage.scheduler-select.name': ['Scheduler', '噪声计划表', '雜訊排程器'],
  'stage.text-to-image.name': ['Text to Image', '文生图', '文生圖'],
  'stage.text-to-speech.name': ['Text to Speech', '语音合成', '文字轉語音'],
  'stage.vae-decode.name':     ['VAE Decode', 'VAE 解码', 'VAE 解碼'],
  'stage.vae-encode.name':     ['VAE Encode', 'VAE 编码', 'VAE 編碼'],
  'stage.temporal-decimation.name': ['Frame Dropper', '视频抽帧', '影片抽幀'],
  'stage.video-to-rgb.name': ['Video → RGB', '视频流 → RGB', '視訊串流 → RGB'],
  'stage.audio-to-pcm.name': ['Audio → PCM', '音频流 → PCM', '音訊串流 → PCM'],
  // sources / sinks / control
  'stage.load-text.name': ['Load Text', '加载文本', '載入文字'],
  'stage.save-text.name': ['Save Text', '保存文本', '儲存文字'],
  'stage.save-audio.name': ['Save Audio', '保存音频', '儲存音訊'],
  'stage.text-input.name': ['Text Input', '文本输入', '文字輸入'],
  'stage.text-prompt.name': ['Text Prompt', '文本提示', '文字提示'],
  'stage.rest-client.name': ['REST Client', 'REST 客户端', 'REST 用戶端'],
  'stage.feedback-rx.name': ['Feedback Rx', '反馈器接收端', '回饋器接收端'],
  'stage.feedback-tx.name': ['Feedback Tx', '反馈器发射端', '回饋器傳送端'],
  'stage.shell.name': ['Shell', 'Shell 命令', 'Shell 指令'],
  'stage.chrono.name': ['Timer', '定时器', '計時器'],
  // audio / video capture + streaming
  'stage.audio-capture.name': ['Audio Capture', '音频采集', '音訊擷取'],
  'stage.rtsp-capture.name': ['RTSP Capture', 'RTSP 采集', 'RTSP 擷取'],
  'stage.hls-broadcast.name': ['HLS Broadcast', 'HLS 广播', 'HLS 廣播'],
  'stage.audio-segment.name': ['Audio Segment (VAD)', '音频分段 (VAD)',
                               '音訊分段 (VAD)'],
  'stage.audio-tagging.name': ['Audio Tagging', '音频标注', '音訊標註'],
  // vision
  'stage.yolo-detection.name': ['YOLO Detector', '目标检测器', '物件偵測器'],
  'stage.byte-track.name': ['ByteTrack', '目标跟踪器', '物件追蹤器'],
  'stage.detection-overlay.name': ['Detection Overlay', '检测结果可视化',
                                   '偵測結果渲染器'],
  'stage.coreml-inference.name': ['CoreML Model', 'CoreML 模型', 'CoreML 模型'],
  // model management
  'stage.model-fetch.name': ['Model Fetch', '模型下载', '模型下載'],
  'stage.model-quantize.name': ['Model Quantize', '模型量化', '模型量化'],
  'stage.model-eval.name': ['Model Eval', '模型评估', '模型評估'],
  'stage.model-benchmark.name': ['Model Benchmark', '模型测速', '模型測速'],
  'stage.lora-fuse.name': ['LoRA Fuse', 'LoRA 融合', 'LoRA 融合'],
  // misc
  'stage.onvif-discovery.name': ['ONVIF Discovery', 'ONVIF 摄像头搜索',
                                 'ONVIF 攝影機搜尋'],
  'stage.videos-db-cleanup.name': ['Video Cleanup', '视频清理', '影片清理'],

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
  'cfg.realtime-vqa.scene_overlap': ['re-issue the previous scene\'s last '
      + 'frame as the next scene\'s first frame, but only across temporally-'
      + 'continuous scenes; false disables it',
      '将上一场景的最后一帧作为下一场景的第一帧重新加入，但仅限时间上连续的场景；'
      + '设为 false 可禁用。',
      '將上一場景的最後一幀作為下一場景的第一幀重新加入，但僅限時間上連續的場景；'
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
