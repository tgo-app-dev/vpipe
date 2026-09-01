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
  'fs.preview_toggle': ['Show / hide preview', '显示 / 隐藏预览',
                        '顯示 / 隱藏預覽'],
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
  'fb.modified':      ['Modified {t}', '修改于 {t}', '修改於 {t}'],
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
  'fb.resize_cols':   ['Drag to resize the columns (double-click to reset)',
                       '拖动以调整列宽（双击恢复默认）',
                       '拖曳以調整欄寬（雙擊恢復預設）'],
  // Preview additions: the .vpipeline stage summary, and the dialog's
  // side-panel toggle.
  'fb.pl_stages':     ['{n} stages', '{n} 个流水线级', '{n} 個階段'],
  'fb.pl_unparsed':   ['Could not parse this pipeline — showing the raw '
                       + 'file.', '无法解析此流水线 — 显示原始文件。',
                       '無法解析此管線 — 顯示原始檔案。'],

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
  'settings.wired_pool':    ['Unswappable memory (wired pool)',
      '不可交换内存（锁定池）', '不可交換記憶體（鎖定池）'],
  'settings.wired_pool_desc': ['Ceiling on memory this process makes '
      + 'unswappable: model weights, a streaming model\u2019s resident '
      + 'blocks, activation scratch. Wired pages cannot be compressed or '
      + 'swapped, so this is what the process genuinely keeps. It can be '
      + 'RAISED while a pipeline runs, but not lowered \u2014 giving wired '
      + 'bytes back means unwiring buffers a model is still reading. 0 '
      + 'restores the default share of RAM.',
      '本进程可锁定为不可交换的内存上限：模型权重、流式模型常驻的层块、'
      + '激活暂存区。锁定的页面无法被压缩或换出，因此这是本进程真正持有的内存。'
      + '流水线运行期间只能调高，不能调低——归还已锁定的内存意味着解锁模型'
      + '正在读取的缓冲区。填 0 则恢复按内存比例的默认值。',
      '本行程可鎖定為不可交換的記憶體上限：模型權重、串流模型常駐的層塊、'
      + '啟動暫存區。鎖定的頁面無法被壓縮或換出，因此這是本行程真正持有的'
      + '記憶體。管線執行期間只能調高，不能調低——歸還已鎖定的記憶體意味著'
      + '解鎖模型正在讀取的緩衝區。填 0 則恢復按記憶體比例的預設值。'],
  'settings.mb':            ['MB', 'MB', 'MB'],
  'settings.wired_updated': ['Wired pool updated', '锁定池已更新',
      '鎖定池已更新'],
  'settings.wired_current': ['current {n} MB · {used} MB wired now',
      '当前 {n} MB · 已锁定 {used} MB', '目前 {n} MB · 已鎖定 {used} MB'],
  'settings.wired_capped':  ['capped at {max} MB by what the GPU can keep '
      + 'resident', '受 GPU 可常驻内存限制，上限为 {max} MB',
      '受 GPU 可常駐記憶體限制，上限為 {max} MB'],
  // Shown whether or not the cap is currently binding: the limit above
  // cannot be read without it. 48 GB is roomy or already clamped
  // depending on this number.
  'settings.wired_device_max': ['GPU maximum {max} MB',
      'GPU 上限 {max} MB', 'GPU 上限 {max} MB'],
  'settings.wired_sysctl':  ['The GPU maximum is a hard ceiling: a larger '
      + 'figure is accepted and clamped to it. To raise it, '
      + 'sudo sysctl iogpu.wired_limit_mb=N (N in MB; 0 restores the '
      + 'default, and it does not survive a reboot).',
      'GPU 上限是硬性上限：填入更大的数值会被自动截断到该上限。如需提高，'
      + '可执行 sudo sysctl iogpu.wired_limit_mb=N（N 以 MB 为单位；填 0 '
      + '恢复默认值，且重启后失效）。',
      'GPU 上限是硬性上限：填入更大的數值會被自動截斷到該上限。如需提高，'
      + '可執行 sudo sysctl iogpu.wired_limit_mb=N（N 以 MB 為單位；填 0 '
      + '恢復預設值，且重新開機後失效）。'],
  'settings.wired_running': ['a pipeline is running \u2014 the limit can be '
      + 'raised but not lowered',
      '流水线正在运行——上限只能调高，不能调低',
      '管線正在執行——上限只能調高，不能調低'],
  'settings.wired_off':     ['wiring is off; nothing is protected from the '
      + 'compressor', '未启用锁定；没有内存受压缩器保护',
      '未啟用鎖定；沒有記憶體受壓縮器保護'],
  'settings.limit_nonneg':  ['limit must be 0 or a positive integer',
      '上限必须为 0 或正整数', '上限必須為 0 或正整數'],
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
  'app.dashboard':    ['Dashboard', '控制台', '儀表板'],
  'app.coming_soon':  ['Coming soon.', '即将推出。', '即將推出。'],

  // ---- Phone shell (js/phone/) ----
  'phone.menu':          ['Menu', '菜单', '選單'],
  'phone.views':         ['Views', '视图', '檢視'],
  'phone.preview':       ['Preview', '预览', '預覽'],
  'phone.compare':       ['Compare', '图像对比', '影像對比'],
  'phone.no_stage_view': ['This server offers no "{id}" panel.',
      '此服务器未提供“{id}”面板。', '此伺服器未提供「{id}」面板。'],
  'phone.desktop_site':  ['Desktop layout', '桌面版界面', '桌面版介面'],
  'phone.desktop_hint':  ['The full dashboard: profiler, database, '
      + 'files, composer and pipeline editing.',
      '完整控制台：性能分析、数据库、文件、自定义与流水线编辑。',
      '完整儀表板：效能分析、資料庫、檔案、自訂與管線編輯。'],
  'phone.desktop_confirm': ['It is built for a large screen and will be '
      + 'hard to use here. The page reloads; come back with the '
      + '"Phone layout" button in its top bar.',
      '该界面为大屏幕设计，在手机上很难操作。页面将重新加载；'
      + '可用其顶栏的“手机版界面”按钮返回。',
      '該介面為大螢幕設計，在手機上很難操作。頁面將重新載入；'
      + '可用其頂列的「手機版介面」按鈕返回。'],
  'phone.desktop_switch': ['Switch', '切换', '切換'],
  'phone.system':        ['System', '系统资源', '系統資源'],
  'phone.sys_window':    ['Last {s} seconds · updated every second',
      '最近 {s} 秒 · 每秒更新', '最近 {s} 秒 · 每秒更新'],
  'phone.phone_layout':  ['Phone layout', '手机版界面', '手機版介面'],
  'phone.pick_pipeline': ['Choose a pipeline', '选择流水线', '選擇管線'],
  'phone.topology_note': ['Stage wiring is edited in the desktop layout.',
      '流水线级的连线请在桌面版中编辑。',
      '管線階段的連線請在桌面版中編輯。'],
  'phone.more':          ['More actions', '更多操作', '更多操作'],
  'phone.save_as':       ['Save as', '另存为', '另存為'],
  'phone.busy_launch':   ['starting', '启动中', '啟動中'],
  'phone.busy_pause':    ['pausing', '暂停中', '暫停中'],
  'phone.busy_stop':     ['stopping', '停止中', '停止中'],
  'phone.load_title':    ['Load pipeline', '加载流水线', '載入管線'],
  'phone.loaded':        ['Loaded {id}', '已加载 {id}', '已載入 {id}'],
  'phone.unloaded':      ['Unloaded {id}', '已卸载 {id}', '已卸載 {id}'],
  'phone.about':         ['About this stage', '流水线级说明', '階段說明'],
  'phone.connections':   ['Connections', '连接', '連接'],
  'phone.inputs':        ['Inputs', '输入端口', '輸入埠'],
  'phone.outputs':       ['Outputs', '输出端口', '輸出埠'],
  'phone.config':        ['Configuration', '配置', '配置'],
  'phone.none':          ['none', '无', '無'],
  'phone.unconnected':   ['not connected', '未连接', '未連接'],
  'phone.no_config':     ['No configurable attributes',
      '没有可配置的属性', '沒有可設定的屬性'],
  'phone.stop_to_edit':  ['Stop the pipeline to edit its configuration.',
      '停止流水线后才能编辑配置。', '停止管線後才能編輯設定。'],
  'phone.cfg_default':   ['default ({v})', '默认（{v}）', '預設（{v}）'],
  'phone.cfg_unset_title': ['Clear — use the stage default',
      '清除 — 使用流水线级默认值', '清除 — 使用階段預設值'],
  'phone.cfg_bad_json':  ['{key}: not valid JSON', '{key}：不是有效的 JSON',
      '{key}：不是有效的 JSON'],
  'phone.cfg_bad_number':['{key}: not a number', '{key}：不是数字',
      '{key}：不是數字'],
  'phone.cfg_applied':   ['{id} configuration applied', '{id} 配置已应用',
      '{id} 設定已套用'],
  'phone.cfg_apply_failed': ['Apply failed: {msg}', '应用失败：{msg}',
      '套用失敗：{msg}'],
  'phone.pick_path':     ['Choose a path', '选择路径', '選擇路徑'],
  'phone.pick_model':    ['Choose a model', '选择模型', '選擇模型'],
  'phone.no_models':     ['No installed models match this field.',
      '没有符合此字段的已安装模型。', '沒有符合此欄位的已安裝模型。'],
  'phone.back':          ['Back', '返回', '返回'],
  'phone.open_raw':      ['Open original', '打开原文件', '開啟原始檔'],
  'phone.no_preview_kind': ['No preview for this file type.',
      '此文件类型无法预览。', '此檔案類型無法預覽。'],
  'phone.text_truncated': ['Showing the first {shown} of {total}.',
      '仅显示前 {shown}，共 {total}。', '僅顯示前 {shown}，共 {total}。'],

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
  'pl.narrow_note':    ['Pane too narrow for the canvas — widen it to '
      + 'wire stages. Tap a stage to edit its configuration here.',
      '窗格过窄，无法显示画布 — 请加宽以连接流水线级。点按某一级即可在此编辑其配置。',
      '窗格過窄，無法顯示畫布 — 請加寬以連接階段。點按某一階段即可在此編輯其設定。'],
  // Same fallback, but the editor is still wide enough to keep its
  // configuration PANE -- so the note points there instead of inline.
  'pl.narrow_note_pane': ['Pane too narrow for the canvas — widen it to '
      + 'wire stages. Selecting one still edits its configuration.',
      '窗格过窄，无法显示画布 — 请加宽以连接流水线级。选中某一级仍可编辑其配置。',
      '窗格過窄，無法顯示畫布 — 請加寬以連接階段。選取某一階段仍可編輯其設定。'],
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
  // Prefixes a toolbox section named after the plugin that contributed
  // the stages under it; the plugin's own name follows, untranslated.
  'pl.plugin_group':   ['Plugin', '插件', '外掛'],
  'nav.plugins':       ['Plugins', '插件', '外掛'],
  'plugins.title':     ['Plugins', '插件', '外掛'],
  'plugins.root':      ['Plugins directory', '插件目录', '外掛目錄'],
  'plugins.root_missing': [
    'This directory does not exist yet. Create it and put plugin dylibs in it.',
    '此目录尚不存在。请创建它并将插件动态库放入其中。',
    '此目錄尚不存在。請建立它並將外掛動態庫放入其中。'],
  // The single most important sentence in this panel: Disable is not an
  // unload, and a user who assumes otherwise will expect memory back.
  'plugins.no_unload': [
    'Plugins are never unloaded: a loaded plugin stays mapped for the life '
    + 'of the process. Disable only stops its stages being offered. Restart '
    + 'to remove one.',
    '插件永不卸载：已加载的插件在进程存续期间始终驻留内存。停用仅停止提供其阶段。'
    + '需重启才能真正移除。',
    '外掛永不卸載：已載入的外掛在行程存續期間始終駐留記憶體。停用僅停止提供其階段。'
    + '需重新啟動才能真正移除。'],
  'plugins.loaded':    ['Loaded', '已加载', '已載入'],
  'plugins.available': ['Available', '可用', '可用'],
  'plugins.none_loaded':    ['No plugins loaded', '未加载插件', '未載入外掛'],
  'plugins.none_available': ['No plugins found', '未找到插件', '未找到外掛'],
  'plugins.load':      ['Load', '加载', '載入'],
  'plugins.loaded_ok': ['Plugin loaded', '插件已加载', '外掛已載入'],
  'plugins.enable':    ['Enable', '启用', '啟用'],
  'plugins.disable':   ['Disable', '停用', '停用'],
  'plugins.disabled_resident': [
    '(disabled, still resident)', '（已停用，仍驻留内存）', '（已停用，仍駐留記憶體）'],
  'plugins.stages':    ['stages', '阶段', '階段'],
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
  'pl.add_stage':      ['Add stage', '添加阶段', '新增階段'],
  // Graph canvas controls + the two labels drawn on a node itself.
  'pl.zoom_out':       ['Zoom out', '缩小', '縮小'],
  'pl.zoom_in':        ['Zoom in', '放大', '放大'],
  'pl.actual_size':    ['Actual size', '实际大小', '實際大小'],
  'pl.fit_all':        ['Fit whole pipeline', '适配整条流水线', '符合整條管線'],
  'pl.center':         ['Center', '居中', '置中'],
  'pl.add_input':      ['add input', '添加输入端口', '新增輸入埠'],
  'pl.needs_config':   ['Needs configuration: {msg}', '需要配置：{msg}',
      '需要設定：{msg}'],
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

  // ---- Connection editor (lane-graph node; phone + narrow desktop) ----
  'conn.title':        ['Connections — {id}', '连接 — {id}', '連接 — {id}'],
  'conn.edit_hint':    ["Edit this stage's connections", '编辑此阶段的连接',
      '編輯此階段的連接'],
  'conn.pick_iport':   ['Which input?', '选择输入端口', '選擇輸入埠'],
  'conn.pick_source':  ['Feed {port} from…', '{port} 的来源…',
      '{port} 的來源…'],
  'conn.back':         ['Inputs', '输入', '輸入'],
  'conn.unconnected':  ['not connected', '未连接', '未連接'],
  'conn.current':      ['current source', '当前来源', '目前來源'],
  'conn.disconnect':   ['Disconnect this input', '断开此输入', '斷開此輸入'],
  'conn.no_iports':    ['This stage has no inputs.', '此阶段没有输入端口。',
      '此階段沒有輸入埠。'],
  'conn.no_sources':   ['No other stage has a compatible output.',
      '没有其他阶段提供兼容的输出。', '沒有其他階段提供相容的輸出。'],
  'conn.incompatible_type': ['needs {want}, emits {got}',
      '需要 {want}，输出 {got}', '需要 {want}，輸出 {got}'],
  'conn.incompatible_tags': ['payload tags do not match',
      '负载标签不匹配', '負載標籤不相符'],
  'conn.stop_to_edit': ['Stop the pipeline to change connections.',
      '停止流水线后才能修改连接。', '停止管線後才能修改連接。'],
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
  'pl.mb_search':      ['Filter by name, variant or type…',
      '按名称、变体或类型筛选…', '依名稱、變體或類型篩選…'],
  'pl.mb_in':          ['takes', '输入', '輸入'],
  'pl.mb_out':         ['makes', '输出', '輸出'],
  'pl.mb_count':       ['{n} models', '{n} 个模型', '{n} 個模型'],
  'pl.mb_count_filtered': ['{n} of {total} models',
      '{total} 个模型中的 {n} 个', '{total} 個模型中的 {n} 個'],
  'pl.mb_no_match':    ['Nothing matches this filter. Clear it to see the '
      + 'compatible models again.',
      '没有符合此筛选条件的模型。清除筛选可重新查看兼容模型。',
      '沒有符合此篩選條件的模型。清除篩選可重新查看相容模型。'],
  'pl.btn_unset':      ['⊘ unset', '⊘ 未设置', '⊘ 未設定'],
  'pl.btn_clear':      ['× clear', '× 清除', '× 清除'],
  'pl.omit_field':     ['omit this field from the config',
      '从配置中省略此字段', '從配置中省略此欄位'],
  'pl.clear_omit':     ['clear (omit this field from the config)',
      '清除（从配置中省略此字段）', '清除（從配置中省略此欄位）'],
  // Free-text fields are tri-state: omitted, present-but-empty, or a value.
  // This placeholder marks the middle one -- the box looks empty either way,
  // so it has to say which is being sent.
  'pl.empty_value':    ['(empty value)', '（空值）', '（空值）'],
  'pl.unset_type_value': ['unset (type a value to set it)',
      '未设置（输入值以设置）', '未設定（輸入值以設定）'],
  'pl.set_empty':      ['send an explicit empty value instead of omitting',
      '发送显式空值而不是省略', '傳送顯式空值而非省略'],
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
  'io.session_log':    ['Session Log', '会话日志', '工作階段日誌'],
  'io.new_view':       ['New view', '新建视图', '新增檢視'],
  'io.add_view':       ['Add a view', '添加一个视图', '新增一個檢視'],
  'io.more_soon':      ['More view types coming soon.',
      '更多视图类型即将推出。', '更多檢視類型即將推出。'],
  'io.split_v':        ['Split vertically (left / right)',
      '垂直拆分（左 / 右）', '垂直分割（左 / 右）'],
  'io.split_h':        ['Split horizontally (top / bottom)',
      '水平拆分（上 / 下）', '水平分割（上 / 下）'],
  // The Shift variants: same split, this panel keeps the OTHER half.
  'io.split_v_alt':    ['Split vertically (this view right)',
      '垂直拆分（本视图在右）', '垂直分割（本檢視在右）'],
  'io.split_h_alt':    ['Split horizontally (this view bottom)',
      '水平拆分（本视图在下）', '水平分割（本檢視在下）'],
  'io.replace_view':   ['Replace view…', '替换视图…', '替換檢視…'],
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
  'composer.replace_view': ['Replace view…', '替换视图…', '替換檢視…'],
  'composer.close':     ['Close', '关闭', '關閉'],
  'composer.no_pipeline': ['No pipeline available', '没有可用的流水线',
                           '沒有可用的管線'],
  'composer.no_saved':  ['No saved layout for a pipeline', '没有已保存的布局',
                         '沒有已儲存的版面'],
  'composer.layout_filter': ['Composer layout (*.json)',
      '组合布局 (*.json)', '組合版面 (*.json)'],
  'composer.save_file_title': ['Save composer layout', '保存组合布局',
                               '儲存組合版面'],
  'composer.load_file_title': ['Load composer layout', '加载组合布局',
                               '載入組合版面'],
  'composer.file_too_large': ['File too large', '文件过大', '檔案過大'],
  'composer.save_pl_title': ['Save pipeline + layout — {id}',
      '保存流水线 + 布局 — {id}', '儲存管線 + 版面 — {id}'],
  'composer.saved_pl_path': ['Saved to {path}', '已保存到 {path}',
                             '已儲存至 {path}'],
  'composer.save_pl_failed': ['Save failed: {msg}', '保存失败：{msg}',
                              '儲存失敗：{msg}'],
  'composer.no_saved_pl': ['No saved arrangement for "{id}"',
      '“{id}”没有已保存的布局', '「{id}」沒有已儲存的版面'],
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
  'userio.interrupt':  ['Interrupt', '中断', '中斷'],
  'userio.interrupt_title': ['Stop the work running right now (e.g. a '
      + 'model’s reply) without ending the pipeline; whatever was '
      + 'produced so far is kept',
      '停止当前正在进行的工作（例如模型正在生成的回复），但不结束流水线；'
      + '已生成的内容会保留',
      '停止目前正在進行的工作（例如模型正在生成的回覆），但不結束管線；'
      + '已產生的內容會保留'],
  'userio.interrupted':['Interrupted {n} running task(s)',
      '已中断 {n} 个正在运行的任务', '已中斷 {n} 個正在執行的工作'],
  'userio.interrupt_idle': ['Nothing running to interrupt',
      '当前没有可中断的任务', '目前沒有可中斷的工作'],
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

  // ---- Stage-provided views (see stage-views.js) ----
  // A view's own strings ship inside its module and are merged at load
  // time; only the host's own failure message lives here.
  'views.load_failed': ['Could not load the "{id}" view: {msg}',
      '无法加载“{id}”视图：{msg}', '無法載入「{id}」檢視：{msg}'],

  // ---- Status bar ----
  'status.ane':          ['ANE', 'ANE', 'ANE'],
  'status.gpu':          ['GPU', 'GPU', 'GPU'],
  'status.gpu_mem':      ['GPU mem', '显存', '顯示記憶體'],
  'status.mem':          ['MEM', '内存', '記憶體'],
  'status.machine_title':['GPU / chip model', 'GPU / 芯片型号', 'GPU / 晶片型號'],
  'status.progress':     ['WORK', '进度', '進度'],
  'status.no_progress':  ['Nothing in progress', '暂无进行中的任务',
      '目前沒有進行中的工作'],
  'status.elapsed':      ['{t} elapsed', '已用 {t}', '已用 {t}'],
  'status.eta_left':     ['{t} left', '剩余 {t}', '剩餘 {t}'],

  // ---- Stage spec docs (overlay; see tOr) ----------------------------
  // These translate the text authored in the C++ stage specs -- the stage
  // display name and doc, every config key's doc, every port's doc -- which
  // reaches the browser in English over /api/stage-types.
  //
  // The en-us slot is EMPTY on purpose. tOr() falls back to the string the
  // API sent, which IS the C++ doc, so an English reader always sees the
  // spec itself and a copy here could only go stale. It measurably did:
  // when this table held English, 7 of its 34 spec entries had drifted from
  // the source they were copied from, some by a whole clause. Leave the
  // slot empty; translate the other two.
  //
  // Keys: cat.<category>, stage.<type>.name, stage.<type>.doc,
  // cfg.<type>.<key>, port.<type>.<port>. A stage whose input and output
  // ports share a name qualifies them, port.<type>.in.<port> /
  // port.<type>.out.<port>, which the port lookup prefers (graph.js).
  //
  // cat.* keeps its English: its fallback is the raw category slug.

  // Stage categories.
  'cat.audio': ['Audio', '音频', '音訊'],
  'cat.control': ['Control', '控制', '控制'],
  'cat.database': ['Database', '数据库', '資料庫'],
  'cat.generative': ['Generative', '生成', '生成'],
  'cat.generic': ['Generic', '通用', '通用'],
  'cat.model-specific-config': ['Model Config', '模型配置', '模型設定'],
  'cat.network': ['Network', '网络', '網路'],
  'cat.preparation': ['Preparation', '准备', '準備'],
  'cat.text': ['Text', '文本', '文字'],
  'cat.vision': ['Vision', '视觉', '視覺'],
  'cat.visual': ['Visual', '影像', '影像'],

  // ---- Audio Capture (audio) ----
  'stage.audio-capture.name': ['', '音频采集', '音訊擷取'],
  'stage.audio-capture.doc': ['',
      '源：通过 FFmpeg avfoundation 采集麦克风 / 线路输入设备，每个音频数据'
      + '包发出一个 EncodedSegment。仅 Apple 平台。无输入端口。',
      '來源：透過 FFmpeg avfoundation 擷取麥克風 / 線路輸入裝置，每個音訊封'
      + '包發出一個 EncodedSegment。僅 Apple 平台。無輸入埠。'],
  'cfg.audio-capture.device_id': ['',
      'avfoundation 设备索引（与 device_name 互斥）',
      'avfoundation 裝置索引（與 device_name 互斥）'],
  'cfg.audio-capture.device_name': ['',
      'avfoundation 设备名称；不区分大小写的子串匹配（与 device_id 互斥）',
      'avfoundation 裝置名稱；不區分大小寫的子字串比對（與 device_id 互斥）'],
  'cfg.audio-capture.sample_rate': ['',
      '请求的输入采样率（0 = 设备默认）',
      '請求的輸入取樣率（0 = 裝置預設）'],
  'cfg.audio-capture.channels': ['',
      '请求的输入声道数（0 = 设备默认）',
      '請求的輸入聲道數（0 = 裝置預設）'],
  'cfg.audio-capture.reconnect_delay_ms': ['',
      '出错后重新打开前的退避时间（毫秒）',
      '出錯後重新開啟前的退避時間（毫秒）'],
  'cfg.audio-capture.oport_depth': ['',
      '输出环形缓冲深度（丢弃最旧）',
      '輸出環形緩衝深度（丟棄最舊）'],
  'port.audio-capture.audio': ['',
      '每个音频数据包一个 EncodedSegment（原始 PCM，设备原生采样率）；下游'
      + '接 audio-to-pcm',
      '每個音訊封包一個 EncodedSegment（原始 PCM，裝置原生取樣率）；下游接 '
      + 'audio-to-pcm'],

  // ---- Audio Segment (VAD) (audio) ----
  'stage.audio-segment.name': ['', '音频分段 (VAD)', '音訊分段 (VAD)'],
  'stage.audio-segment.doc': ['',
      'Silero-VAD 语音分段器：在单声道 PCM 流上做窗口/步长的语音活动检测，'
      + '配合一个带迟滞的有限状态机，发出 [start_us,end_us) 的语段标记，供'
      + '下游切片器使用（例如流式的 audio-transcribe）。输出端口上不流动 '
      + 'PCM。',
      'Silero-VAD 語音分段器：在單聲道 PCM 串流上做視窗/跳距的語音活動偵測'
      + '，配合一個帶遲滯的有限狀態機，發出 [start_us,end_us) 的語段標記，'
      + '供下游切片器使用（例如串流的 audio-transcribe）。輸出埠上不流動 '
      + 'PCM。'],
  'cfg.audio-segment.model_path': ['',
      'Silero VAD 模型：models 数据库中的键（由 model-fetch 注册）或一个 '
      + '.mlpackage / .mlmodelc 目录；同名时数据库键优先于路径',
      'Silero VAD 模型：models 資料庫中的鍵（由 model-fetch 註冊）或一個 '
      + '.mlpackage / .mlmodelc 目錄；同名時資料庫鍵優先於路徑'],
  'cfg.audio-segment.compute_units': ['',
      '0=仅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE',
      '0=僅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE'],
  'cfg.audio-segment.sample_rate': ['',
      '预期的输入采样率（Hz），[8000,48000]',
      '預期的輸入取樣率（Hz），[8000,48000]'],
  'cfg.audio-segment.window_samples': ['',
      '每次推理的输入长度，[32,1048576]',
      '每次推理的輸入長度，[32,1048576]'],
  'cfg.audio-segment.hop_samples': ['',
      '每次推理前进多少采样，[1,window_samples]',
      '每次推理前進多少取樣，[1,window_samples]'],
  'cfg.audio-segment.speech_threshold': ['',
      'VAD 概率大于等于该值即进入候选语音态',
      'VAD 機率大於等於該值即進入候選語音態'],
  'cfg.audio-segment.silence_threshold': ['',
      'VAD 概率小于该值即进入候选静音态（须小于 speech_threshold）',
      'VAD 機率小於該值即進入候選靜音態（須小於 speech_threshold）'],
  'cfg.audio-segment.min_speech_ms': ['',
      '候选语音持续多少毫秒后才开启一个语段',
      '候選語音持續多少毫秒後才開啟一個語段'],
  'cfg.audio-segment.min_silence_ms': ['',
      '候选静音持续多少毫秒后才关闭一个语段',
      '候選靜音持續多少毫秒後才關閉一個語段'],
  'cfg.audio-segment.max_segment_s': ['',
      '超过该时长即强制关闭（标记为不完整），(0,120]',
      '超過該長度即強制關閉（標記為不完整），(0,120]'],
  'cfg.audio-segment.pre_pad_ms': ['',
      '把语段起点往前拉多少毫秒',
      '把語段起點往前拉多少毫秒'],
  'cfg.audio-segment.post_pad_ms': ['',
      '把语段终点往后推多少毫秒',
      '把語段終點往後推多少毫秒'],
  'cfg.audio-segment.input_feature_name': ['',
      'PCM 输入特征名',
      'PCM 輸入特徵名'],
  'cfg.audio-segment.prob_feature_name': ['',
      'VAD 概率输出特征名',
      'VAD 機率輸出特徵名'],
  'cfg.audio-segment.state_h_in_name': ['',
      'LSTM 隐状态输入名',
      'LSTM 隱狀態輸入名'],
  'cfg.audio-segment.state_c_in_name': ['',
      'LSTM 细胞状态输入名',
      'LSTM 記憶單元狀態輸入名'],
  'cfg.audio-segment.state_h_out_name': ['',
      '下一步的 LSTM 隐状态输出名',
      '下一步的 LSTM 隱狀態輸出名'],
  'cfg.audio-segment.state_c_out_name': ['',
      '下一步的 LSTM 细胞状态输出名',
      '下一步的 LSTM 記憶單元狀態輸出名'],
  'cfg.audio-segment.sr_feature_name': ['',
      'int32 采样率输入特征名；留空则禁用',
      'int32 取樣率輸入特徵名；留空則停用'],
  'cfg.audio-segment.state_h_shape': ['',
      'LSTM 隐状态张量形状（例如 [1,128] 或 [2,1,64]）',
      'LSTM 隱狀態張量形狀（例如 [1,128] 或 [2,1,64]）'],
  'cfg.audio-segment.state_c_shape': ['',
      'LSTM 细胞状态张量形状（例如 [1,128] 或 [2,1,64]）',
      'LSTM 記憶單元狀態張量形狀（例如 [1,128] 或 [2,1,64]）'],
  'port.audio-segment.audio': ['',
      '单声道 F32 PCM TensorBeat [N] 或 [1,N]；语段时间由 '
      + 'sideband.timestamp_us 驱动',
      '單聲道 F32 PCM TensorBeat [N] 或 [1,N]；語段時間由 '
      + 'sideband.timestamp_us 驅動'],
  'port.audio-segment.segments': ['',
      '每关闭一个语段发出一个 FlexData：{start_us, end_us, index, '
      + 'is_partial}；只是时间标记，不含 PCM',
      '每關閉一個語段發出一個 FlexData：{start_us, end_us, index, '
      + 'is_partial}；只是時間標記，不含 PCM'],

  // ---- Audio Tagging (audio) ----
  'stage.audio-tagging.name': ['', '音频标注', '音訊標註'],
  'stage.audio-tagging.doc': ['',
      '在滑动的 PCM 窗口上运行 AudioSet 标注模型（CoreML；按 model_kind 选'
      + '择 CED-base 或 BEATs），每个窗口发出得分最高的 k 个类别标签。',
      '在滑動的 PCM 視窗上執行 AudioSet 標註模型（CoreML；按 model_kind 選'
      + '擇 CED-base 或 BEATs），每個視窗發出得分最高的 k 個類別標籤。'],
  'cfg.audio-tagging.model_path': ['',
      'AudioSet 标注模型：models 数据库中的键（由 model-fetch 注册，例如 '
      + 'BEATs 补充模型）或一个 .mlpackage / .mlmodelc 目录；同名时数据库键'
      + '优先于路径',
      'AudioSet 標註模型：models 資料庫中的鍵（由 model-fetch 註冊，例如 '
      + 'BEATs 補充模型）或一個 .mlpackage / .mlmodelc 目錄；同名時資料庫鍵'
      + '優先於路徑'],
  'cfg.audio-tagging.model_kind': ['',
      '模型族："beats" | "ced"（决定标签表与形状默认值）',
      '模型族："beats" | "ced"（決定標籤表與形狀預設值）'],
  'cfg.audio-tagging.compute_units': ['',
      '0=仅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE',
      '0=僅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE'],
  'cfg.audio-tagging.sample_rate': ['',
      '预期的输入采样率（Hz），[8000,48000]',
      '預期的輸入取樣率（Hz），[8000,48000]'],
  'cfg.audio-tagging.window_seconds': ['',
      '窗口长度（秒），(0,60]（beats：10，ced：5）',
      '視窗長度（秒），(0,60]（beats：10，ced：5）'],
  'cfg.audio-tagging.hop_seconds': ['',
      '每次运行前进多少秒，(0,窗口长度]（beats：8，ced：4）',
      '每次執行前進多少秒，(0,視窗長度]（beats：8，ced：4）'],
  'cfg.audio-tagging.top_k': ['',
      '每个窗口输出多少个标签，>= 1',
      '每個視窗輸出多少個標籤，>= 1'],
  'cfg.audio-tagging.score_threshold': ['',
      '丢弃概率低于该值的标签，[0,1]',
      '丟棄機率低於該值的標籤，[0,1]'],
  'port.audio-tagging.pcm': ['',
      '单声道 F32 16 kHz PCM TensorBeat [N] 或 [1,N]；边带的 timestamp_us/'
      + 'sample_rate 会被采用',
      '單聲道 F32 16 kHz PCM TensorBeat [N] 或 [1,N]；邊帶的 timestamp_us/'
      + 'sample_rate 會被採用'],
  'port.audio-tagging.tags': ['',
      '每个窗口一个 FlexData：得分最高的 top_k 个 AudioSet 标签 {label,'
      + 'index,score} + 窗口元数据',
      '每個視窗一個 FlexData：得分最高的 top_k 個 AudioSet 標籤 {label,'
      + 'index,score} + 視窗中繼資料'],

  // ---- Audio Temporal Resample (audio) ----
  'stage.audio-temporal-resample.name': ['',
      '音频时间重采样',
      '音訊時間重新取樣'],
  'stage.audio-temporal-resample.doc': ['',
      '通过一条 FFmpeg 滤镜链把 PCM 重采样到另一个采样率和/或速度，音高可以'
      + '保持不变、跟随速度变化，或按指定的半音数升高或降低。',
      '透過一條 FFmpeg 濾鏡鏈把 PCM 重新取樣到另一個取樣率和/或速度，音高可'
      + '以保持不變、跟隨速度變化，或按指定的半音數升高或降低。'],
  'cfg.audio-temporal-resample.output_sample_rate': ['',
      '要重采样到的目标采样率——它是时间分辨率，既不改变时长也不改变音高。0'
      + '（默认）保持输入的采样率。这是模型会强制要求的那个旋钮：MiniMax-H3'
      + ' 的音频 VAE 只读 32000，Qwen3-ASR 只读 16000',
      '要重新取樣到的目標取樣率——它是時間解析度，既不改變長度也不改變音高。'
      + '0（預設）保持輸入的取樣率。這是模型會強制要求的那個旋鈕：'
      + 'MiniMax-H3 的音訊 VAE 只讀 32000，Qwen3-ASR 只讀 16000'],
  'cfg.audio-temporal-resample.speed': ['',
      '时长因子：输出的时长是输入的 1/speed。2.0 是两倍快，0.5 是一半。音高'
      + '会怎样是 `pitch` 的事，与这一项无关。默认 1.0',
      '長度因子：輸出的長度是輸入的 1/speed。2.0 是兩倍快，0.5 是一半。音高'
      + '會怎樣是 `pitch` 的事，與這一項無關。預設 1.0'],
  'cfg.audio-temporal-resample.pitch': ['',
      '"maintain"（默认）在时长变化时保持音高——WSOLA 时间伸缩（ffmpeg 的 `'
      + 'atempo`），这正是让人声仍然可辨的做法。"follow" 让音高跟着速度走，'
      + '变成花栗鼠音或拖腔：磁带式行为，而且不费什么——因为它只是一次重采样'
      + '而已。"raise" / "lower" 按 `pitch_semitones` 独立于速度做移调，因'
      + '此两个旋钮可以叠加',
      '"maintain"（預設）在長度變化時保持音高——WSOLA 時間伸縮（ffmpeg 的 `'
      + 'atempo`），這正是讓人聲仍然可辨的做法。"follow" 讓音高跟著速度走，'
      + '變成花栗鼠音或拖腔：磁帶式行為，而且不費什麼——因為它只是一次重新取'
      + '樣而已。"raise" / "lower" 按 `pitch_semitones` 獨立於速度做移調，'
      + '因此兩個旋鈕可以疊加'],
  'cfg.audio-temporal-resample.pitch_semitones': ['',
      '`raise` / `lower` 移调多少，按十二平均律的半音计（12 = 一个八度）。'
      + '对这两者必须为正——方向已经写在 `pitch` 里，这里再用正负号就会出现'
      + '同一个请求的两种写法，还多出一种一不小心互相抵消的可能。maintain /'
      + ' follow 会忽略它',
      '`raise` / `lower` 移調多少，按十二平均律的半音計（12 = 一個八度）。'
      + '對這兩者必須為正——方向已經寫在 `pitch` 裡，這裡再用正負號就會出現'
      + '同一個請求的兩種寫法，還多出一種一不小心互相抵銷的可能。maintain /'
      + ' follow 會忽略它'],
  'cfg.audio-temporal-resample.stacked': ['',
      'false（默认）：每个节拍一段 PCM，输出端口自成一个时钟域，因为速度变'
      + '化并不是一个节拍进、一个节拍出。true：整段波形装在一个节拍里——即 `'
      + 'temporal-stack` 构建的形状——于是本阶段是 1:1 的，输出端口与输入端'
      + '口共用时钟，不跨越任何时钟域。这是声明出来的而不是探测出来的：时钟'
      + '分析在启动时就运行，那时还没有任何节拍存在',
      'false（預設）：每個節拍一段 PCM，輸出埠自成一個時脈域，因為速度變化'
      + '並不是一個節拍進、一個節拍出。true：整段波形裝在一個節拍裡——即 `'
      + 'temporal-stack` 建構的形狀——於是本階段是 1:1 的，輸出埠與輸入埠共'
      + '用時脈，不跨越任何時脈域。這是宣告出來的而不是探測出來的：時脈分析'
      + '在啟動時就執行，那時還沒有任何節拍存在'],
  'port.audio-temporal-resample.in.pcm': ['',
      'f32 PCM TensorBeat，单声道 [N] 或平面的 [C, N]，边带带 `sample_rate`',
      'f32 PCM TensorBeat，單聲道 [N] 或平面的 [C, N]，邊帶帶 `sample_rate`'],
  'port.audio-temporal-resample.out.pcm': ['',
      '同样的布局，位于 `output_sample_rate`，边带中的采样率与时长已重写。'
      + '流式情形下，分块长度由滤镜自己决定。声明的时钟组是流式情形下的答案'
      + '；`stacked` 时 oport_clock_group() 与输入端口共用时钟',
      '同樣的版面，位於 `output_sample_rate`，邊帶中的取樣率與長度已重寫。'
      + '串流情形下，分塊長度由濾鏡自己決定。宣告的時脈組是串流情形下的答案'
      + '；`stacked` 時 oport_clock_group() 與輸入埠共用時脈'],

  // ---- Audio → PCM (audio) ----
  'stage.audio-to-pcm.name': ['', '音频流 → PCM', '音訊串流 → PCM'],
  'stage.audio-to-pcm.doc': ['',
      '解码音频 EncodedSegment 并重采样为 output_sample_rate 上的 F32 PCM（'
      + '单声道，或平面立体声），按固定时长分块发出。跨越“数据包速率 -> 分'
      + '块速率”两个时钟域。',
      '解碼音訊 EncodedSegment 並重新取樣為 output_sample_rate 上的 F32 PCM'
      + '（單聲道，或平面立體聲），按固定長度分塊發出。跨越「封包速率 -> 分'
      + '塊速率」兩個時脈域。'],
  'cfg.audio-to-pcm.output_sample_rate': ['',
      '重采样目标频率（Hz），[1000,384000]',
      '重新取樣目標頻率（Hz），[1000,384000]'],
  'cfg.audio-to-pcm.channels': ['',
      '输出声道数，1（单声道，默认）或 2（立体声）。单声道发出 [N]；立体声'
      + '发出平面的 [2, N]——声道优先，也就是本代码树中所有平面 PCM 消费者读'
      + '取的布局，而不是重采样器内部产生的 LRLR 交织格式。下混为单声道是有'
      + '损的，而这对某些消费者很要紧：以单声道送给 MiniMax-H3 的参考音轨会'
      + '被复制上混回去，于是本来有立体声像的文件，左右两路却完全相同',
      '輸出聲道數，1（單聲道，預設）或 2（立體聲）。單聲道發出 [N]；立體聲'
      + '發出平面的 [2, N]——聲道優先，也就是本程式碼樹中所有平面 PCM 消費者'
      + '讀取的版面，而不是重新取樣器內部產生的 LRLR 交錯格式。下混為單聲道'
      + '是失真的，而這對某些消費者很要緊：以單聲道送給 MiniMax-H3 的參考音'
      + '軌會被複製上混回去，於是本來有立體聲像的檔案，左右兩路卻完全相同'],
  'cfg.audio-to-pcm.chunk_duration_s': ['',
      '每次发出的分块最短秒数，(0,600]',
      '每次發出的分塊最短秒數，(0,600]'],
  'cfg.audio-to-pcm.chunk_overlap_s': ['',
      '每个发出的分块保留多少秒，好让下一个分块以它们开头——这段上下文让消费'
      + '者能看到紧接其前的内容。必须小于 chunk_duration_s。此后每个节拍前'
      + '进 (chunk_duration_s - chunk_overlap_s)，因此 N 个分块覆盖的音频少'
      + '于 N x chunk_duration_s。0（默认）= 相邻分块不共享任何内容',
      '每個發出的分塊保留多少秒，好讓下一個分塊以它們開頭——這段上下文讓消費'
      + '者能看到緊接其前的內容。必須小於 chunk_duration_s。此後每個節拍前'
      + '進 (chunk_duration_s - chunk_overlap_s)，因此 N 個分塊涵蓋的音訊少'
      + '於 N x chunk_duration_s。0（預設）= 相鄰分塊不共用任何內容'],
  'cfg.audio-to-pcm.max_chunk_duration_s': ['',
      '分块秒数的硬上限；须大于等于 chunk_duration_s',
      '分塊秒數的硬上限；須大於等於 chunk_duration_s'],
  'cfg.audio-to-pcm.oport_capacity': ['',
      '输出端口缓冲深度（以分块计），须大于等于 1',
      '輸出埠緩衝深度（以分塊計），須大於等於 1'],
  'cfg.audio-to-pcm.flush_on_eos': ['',
      '源关闭时把不完整的分块也发出去',
      '來源關閉時把不完整的分塊也發出去'],
  'cfg.audio-to-pcm.emit_log_every': ['',
      '每发出第 N 个分块记录一行 INFO 日志（须大于等于 1）',
      '每發出第 N 個分塊記錄一行 INFO 日誌（須大於等於 1）'],
  'port.audio-to-pcm.audio': ['',
      'kind=Audio 的 EncodedSegment（例如来自 rtsp-capture 的 AAC，或原始 '
      + 'PCM）',
      'kind=Audio 的 EncodedSegment（例如來自 rtsp-capture 的 AAC，或原始 '
      + 'PCM）'],
  'port.audio-to-pcm.pcm': ['',
      '位于 output_sample_rate 上的 F32 PCM TensorBeat——单声道为 [N]，'
      + 'channels=2 时为平面的 [2, N]；边带带 ts/sr/duration',
      '位於 output_sample_rate 上的 F32 PCM TensorBeat——單聲道為 [N]，'
      + 'channels=2 時為平面的 [2, N]；邊帶帶 ts/sr/duration'],

  // ---- Transcribe (audio) ----
  'stage.audio-transcribe.name': ['', '语音转录', '語音轉錄'],
  'stage.audio-transcribe.doc': ['',
      '用 Qwen3-ASR 语言模型（编码器 + 贪心/采样解码）转录每段传入的 PCM 音'
      + '频，通过界面委托记录转录文本，并在 0 号输出端口以 FlexData {text[,'
      + ' start_us, end_us]} 的形式发出。',
      '用 Qwen3-ASR 語言模型（編碼器 + 貪婪/取樣解碼）轉錄每段傳入的 PCM 音'
      + '訊，透過介面委派記錄轉錄文字，並在 0 號輸出埠以 FlexData {text[, '
      + 'start_us, end_us]} 的形式發出。'],
  'cfg.audio-transcribe.hf_dir': ['',
      'ASR 语言模型：models 数据库中的键（由 model-fetch 注册的 '
      + 'huggingface.co 路径），或一个文件系统路径。同名时数据库键优先于路'
      + '径。',
      'ASR 語言模型：models 資料庫中的鍵（由 model-fetch 註冊的 '
      + 'huggingface.co 路徑），或一個檔案系統路徑。同名時資料庫鍵優先於路'
      + '徑。'],
  'cfg.audio-transcribe.compute_dtype': ['',
      'bf16 | f16 | f32',
      'bf16 | f16 | f32'],
  'cfg.audio-transcribe.page_tokens': ['',
      'ContextManager 的 K/V 分页大小',
      'ContextManager 的 K/V 分頁大小'],
  'cfg.audio-transcribe.max_pages': ['',
      '每个语言模型的分页池容量（>= 1）',
      '每個語言模型的分頁池容量（>= 1）'],
  'cfg.audio-transcribe.max_new_tokens': ['',
      '每段音频的生成预算（>= 1）',
      '每段音訊的生成預算（>= 1）'],
  'cfg.audio-transcribe.sample_rate': ['',
      '边带没有 sample_rate 时使用的后备采样率（Hz，>= 1）',
      '邊帶沒有 sample_rate 時使用的後備取樣率（Hz，>= 1）'],
  'cfg.audio-transcribe.language_hint': ['',
      '非空时会先发出 \'language X\'，使模型只生成转录文本',
      '非空時會先發出 \'language X\'，使模型只產生轉錄文字'],
  'cfg.audio-transcribe.pcm_buffer_s': ['',
      '流式模式下滚动 PCM 缓冲区的长度（秒），(0,300]',
      '串流模式下滾動 PCM 緩衝區的長度（秒），(0,300]'],
  'cfg.audio-transcribe.late_marker_skip': ['',
      '丢弃起点已早于缓冲区范围的分段标记',
      '丟棄起點已早於緩衝區範圍的分段標記'],
  'port.audio-transcribe.audio': ['',
      '单声道 F32 PCM TensorBeat [N] 或 [1,N]；块模式下每个节拍一段音频，流'
      + '式模式下是连续的 PCM 流；sideband.sample_rate 会被采用',
      '單聲道 F32 PCM TensorBeat [N] 或 [1,N]；區塊模式下每個節拍一段音訊，'
      + '串流模式下是連續的 PCM 串流；sideband.sample_rate 會被採用'],
  'port.audio-transcribe.segments': ['',
      '可选的 FlexData 语段标记 {start_us,end_us,index,is_partial}（例如来'
      + '自 audio-segment）；接上它会把本阶段切换到流式模式（按标记从滚动 '
      + 'PCM 缓冲区中切片）',
      '可選的 FlexData 語段標記 {start_us,end_us,index,is_partial}（例如來'
      + '自 audio-segment）；接上它會把本階段切換到串流模式（按標記從滾動 '
      + 'PCM 緩衝區中切片）'],
  'port.audio-transcribe.sampler': ['',
      '可选的 token 采样器配置 FlexData（sampler-select）。在第一个节拍时锁'
      + '存，之后每段音频都沿用；未接线 = 贪心（argmax）解码',
      '可選的 token 取樣器組態 FlexData（sampler-select）。在第一個節拍時鎖'
      + '存，之後每段音訊都沿用；未接線 = 貪婪（argmax）解碼'],
  'port.audio-transcribe.transcript': ['',
      '每段转录完成的音频一个 FlexData {text, start_us, end_us}（start_us/'
      + 'end_us 仅在流式模式下出现）。可选——不接也没问题（转录文本仍会记录'
      + '）。',
      '每段轉錄完成的音訊一個 FlexData {text, start_us, end_us}（start_us/'
      + 'end_us 僅在串流模式下出現）。可選——不接也沒問題（轉錄文字仍會記錄'
      + '）。'],

  // ---- Load Audio (audio) ----
  'stage.load-audio.name': ['', '加载音频', '載入音訊'],
  'stage.load-audio.doc': ['',
      '源：解复用音频文件或 URL，逐个节拍发出其编码数据包——相当于 '
      + 'rtsp-capture 音频输出端口的文件版。下游接 audio-to-pcm，由它解码并'
      + '重采样。',
      '來源：解多工音訊檔案或 URL，逐個節拍發出其編碼封包——相當於 '
      + 'rtsp-capture 音訊輸出埠的檔案版。下游接 audio-to-pcm，由它解碼並重'
      + '新取樣。'],
  'cfg.load-audio.input_url': ['',
      '音频文件路径或网络 URL。所安装的 FFmpeg 能解复用的任何容器与编码格式'
      + '——mp3、m4a/aac、flac、wav、ogg、opus，或视频文件中的音频轨',
      '音訊檔案路徑或網路 URL。所安裝的 FFmpeg 能解多工的任何容器與編碼格式'
      + '——mp3、m4a/aac、flac、wav、ogg、opus，或影片檔案中的音訊軌'],
  'cfg.load-audio.stream_index': ['',
      '要读取的流的绝对索引（按 ffprobe 的编号，包含视频流在内），而不是第 '
      + 'N 条音频流；-1（默认）取文件中的第一条音频流',
      '要讀取的串流絕對索引（按 ffprobe 的編號，包含視訊串流在內），而不是'
      + '第 N 條音訊串流；-1（預設）取檔案中的第一條音訊串流'],
  'cfg.load-audio.start_s': ['',
      '读取第一个数据包前先定位到该媒体时间（自文件开头起的秒数）；0（默认'
      + '）从头开始。精度到数据包：第一个采样与请求位置的偏差不超过一个数据'
      + '包',
      '讀取第一個封包前先定位到該媒體時間（自檔案開頭起的秒數）；0（預設）'
      + '從頭開始。精度到封包：第一個取樣與請求位置的偏差不超過一個封包'],
  'cfg.load-audio.duration_s': ['',
      '在 start_s 之后这么多秒处停止；0（默认）读到文件结尾。跨越终点的那个'
      + '数据包会被完整发出，因此窗口可能多出一个数据包',
      '在 start_s 之後這麼多秒處停止；0（預設）讀到檔案結尾。跨越終點的那個'
      + '封包會被完整發出，因此視窗可能多出一個封包'],
  'cfg.load-audio.read_timeout_ms': ['',
      '网络打开/读取超时，单位毫秒；0 = 不限时',
      '網路開啟/讀取逾時，單位毫秒；0 = 不限時'],
  'cfg.load-audio.options': ['',
      '传给解复用器 open 的额外 av_dict 选项',
      '傳給解多工器 open 的額外 av_dict 選項'],
  'port.load-audio.audio': ['',
      '每个节拍一个编码数据包，形式为 kind=Audio 的 EncodedSegment，携带该'
      + '流的 codec_id / sample_rate / channels / extradata。请接到 '
      + 'audio-to-pcm，由它负责解码与重采样',
      '每個節拍一個編碼封包，形式為 kind=Audio 的 EncodedSegment，攜帶該串'
      + '流的 codec_id / sample_rate / channels / extradata。請接到 '
      + 'audio-to-pcm，由它負責解碼與重新取樣'],

  // ---- Save Audio (audio) ----
  'stage.save-audio.name': ['', '保存音频', '儲存音訊'],
  'stage.save-audio.doc': ['',
      '汇聚节点：把每段传入的 PCM（一个节拍 = 一段音频）编码成文件：经 '
      + 'FFmpeg 输出 AAC / MP3 / M4A，或直接写 16 位 WAV。它是 save-image '
      + '的音频对应物，与 text-to-speech 配套。多个节拍会在扩展名之前追加递'
      + '增编号。',
      '匯聚節點：把每段傳入的 PCM（一個節拍 = 一段音訊）編碼成檔案：經 '
      + 'FFmpeg 輸出 AAC / MP3 / M4A，或直接寫 16 位元 WAV。它是 save-image'
      + ' 的音訊對應物，與 text-to-speech 配套。多個節拍會在副檔名之前附加'
      + '遞增編號。'],
  'cfg.save-audio.output_path': ['',
      '输出文件路径；没有扩展名时会按 `format` 补上一个',
      '輸出檔案路徑；沒有副檔名時會按 `format` 補上一個'],
  'cfg.save-audio.format': ['',
      'wav | aac | mp3 | m4a（m4a => mp4 容器中的 AAC）。默认：从 '
      + 'output_path 的扩展名推断，否则为 wav',
      'wav | aac | mp3 | m4a（m4a => mp4 容器中的 AAC）。預設：從 '
      + 'output_path 的副檔名推斷，否則為 wav'],
  'cfg.save-audio.bitrate': ['',
      'AAC / MP3 的目标比特率（比特/秒）',
      'AAC / MP3 的目標位元率（位元/秒）'],
  'cfg.save-audio.sample_rate': ['',
      '0 = 使用负载中的 sideband.sample_rate；两者都没有时用 24000',
      '0 = 使用負載中的 sideband.sample_rate；兩者都沒有時用 24000'],
  'port.save-audio.pcm': ['',
      'f32 PCM 的 TensorBeat，秩为 1 的 [采样数]（单声道）或秩为 2 的 [声道'
      + '数, 采样数]；sideband.sample_rate 会被采用。每个节拍是一段完整的音'
      + '频。',
      'f32 PCM 的 TensorBeat，秩為 1 的 [取樣數]（單聲道）或秩為 2 的 [聲道'
      + '數, 取樣數]；sideband.sample_rate 會被採用。每個節拍是一段完整的音'
      + '訊。'],
  'port.save-audio.info': ['',
      '可选的 FlexData 对象，描述刚写出的文件：{path, format, samples, '
      + 'sample_rate, duration_s}。下游消费者是可选的。',
      '可選的 FlexData 物件，描述剛寫出的檔案：{path, format, samples, '
      + 'sample_rate, duration_s}。下游消費者是可選的。'],

  // ---- Buffer (control) ----
  'stage.buffer.name': ['', '缓冲寄存器', '緩衝暫存器'],
  'stage.buffer.doc': ['',
      '单节拍采样保持寄存器。`advance` 把下一个 `in` 节拍读入 1 节拍缓冲区'
      + '（丢弃原先保存的内容）；`emit` 把缓冲的节拍复制一份送到 `out`。用'
      + '于解耦上下游的速率：推进快于发出会丢弃输入，发出快于推进则重复保持'
      + '的节拍。',
      '單節拍取樣保持暫存器。`advance` 把下一個 `in` 節拍讀入 1 節拍緩衝區'
      + '（丟棄原先保存的內容）；`emit` 把緩衝的節拍複製一份送到 `out`。用'
      + '於解耦上下游的速率：推進快於發出會丟棄輸入，發出快於推進則重複保持'
      + '的節拍。'],
  'port.buffer.in': ['',
      '任意节拍；每次 advance 触发采样一个进入 1 节拍缓冲区',
      '任意節拍；每次 advance 觸發取樣一個進入 1 節拍緩衝區'],
  'port.buffer.advance': ['',
      '任意节拍；每一个都把下一个 `in` 节拍读入缓冲区，替换（丢弃）此前保持'
      + '的节拍',
      '任意節拍；每一個都把下一個 `in` 節拍讀入緩衝區，取代（丟棄）此前保持'
      + '的節拍'],
  'port.buffer.emit': ['',
      '任意节拍；每一个都把缓冲的节拍复制一份送到 `out`（若尚未缓冲过任何节'
      + '拍，则不发出）',
      '任意節拍；每一個都把緩衝的節拍複製一份送到 `out`（若尚未緩衝過任何節'
      + '拍，則不發出）'],
  'port.buffer.out': ['',
      '缓冲节拍的副本，每次 emit 触发一个',
      '緩衝節拍的副本，每次 emit 觸發一個'],

  // ---- call (control) ----
  'stage.call.doc': ['',
      '把另一条流水线当作函数调用：call 阶段的输入/输出端口连到被调流水线的'
      + '输入/输出端口，运行时在启动时将其内联展开。纯结构性阶段——它本身从'
      + '不运行。',
      '把另一條管線當作函式呼叫：call 階段的輸入/輸出埠連到被呼叫管線的輸入'
      + '/輸出埠，執行階段在啟動時將其內聯展開。純結構性階段——它本身從不執'
      + '行。'],
  'cfg.call.pipeline': ['',
      '要内联的被调流水线 id',
      '要內聯的被呼叫管線 id'],
  'cfg.call.num_oports': ['',
      '输出端口数量；必须等于 callee.num_oports()',
      '輸出埠數量；必須等於 callee.num_oports()'],

  // ---- Timer (control) ----
  'stage.chrono.name': ['', '定时器', '計時器'],
  'stage.chrono.doc': ['',
      '周期性源：以固定速率（frequency_hz 或 period_*）发出 TriggerBeat，可'
      + '限定次数。',
      '週期性來源：以固定速率（frequency_hz 或 period_*）發出 TriggerBeat，'
      + '可限定次數。'],
  'cfg.chrono.frequency_hz': ['',
      '每秒滴答次数；与 period_* 互斥',
      '每秒滴答次數；與 period_* 互斥'],
  'cfg.chrono.period_seconds': ['',
      '周期中的秒数部分（period_* 各项相加）',
      '週期中的秒數部分（period_* 各項相加）'],
  'cfg.chrono.period_minutes': ['', '周期中的分钟部分', '週期中的分鐘部分'],
  'cfg.chrono.period_hours': ['', '周期中的小时部分', '週期中的小時部分'],
  'cfg.chrono.period_days': ['', '周期中的天数部分', '週期中的天數部分'],
  'cfg.chrono.count': ['',
      '发出多少个节拍后结束；0 = 永不停止',
      '發出多少個節拍後結束；0 = 永不停止'],
  'port.chrono.tick': ['', '周期性的 TriggerBeat', '週期性的 TriggerBeat'],

  // ---- Feedback In (control) ----
  'stage.feedback-rx.name': ['', '反馈器接收端', '回饋器接收端'],
  'stage.feedback-rx.doc': ['',
      '单时钟域反馈对的接收端：缓存 iport 上最近的一个节拍，供同名的 '
      + 'feedback-tx 重新发出（相当于延迟一轮的寄存器）。',
      '單一時脈域回饋對的接收端：快取輸入埠上最近的一個節拍，供同名的 '
      + 'feedback-tx 重新發出（相當於延遲一輪的暫存器）。'],
  'port.feedback-rx.in': ['',
      '任意节拍；最新的一个会被缓存，供发射端取用',
      '任意節拍；最新的一個會被快取，供傳送端取用'],

  // ---- Feedback Out (control) ----
  'stage.feedback-tx.name': ['', '反馈器发射端', '回饋器傳送端'],
  'stage.feedback-tx.doc': ['',
      '单时钟域反馈对的源端：等待同名的 feedback-rx 收到新节拍，然后在 0 号'
      + '输出端口重新发出它的副本。',
      '單一時脈域回饋對的來源端：等待同名的 feedback-rx 收到新節拍，然後在 '
      + '0 號輸出埠重新發出它的副本。'],
  'cfg.feedback-tx.from': ['',
      '要中继的 feedback-rx 阶段 id',
      '要轉送的 feedback-rx 階段 id'],
  'cfg.feedback-tx.prime': ['',
      '在任何反馈出现之前先发出什么，好让第一轮没有历史的循环能够启动。\''
      + 'none\'（默认）等待 rx 收到节拍——而当那个节拍正是本阶段发出的结果时'
      + '，就会死锁。\'empty-tensor\' 先发出一个空的 TensorBeatPayload：这是“'
      + '本轮没有内容”的约定写法，video-ref-encoder 等消费者会直接跳过它。'
      + '每次运行只预置一次',
      '在任何回饋出現之前先發出什麼，好讓第一輪沒有歷史的迴圈能夠啟動。\''
      + 'none\'（預設）等待 rx 收到節拍——而當那個節拍正是本階段發出的結果時'
      + '，就會死結。\'empty-tensor\' 先發出一個空的 TensorBeatPayload：這是'
      + '「本輪沒有內容」的約定寫法，video-ref-encoder 等消費者會直接跳過它'
      + '。每次執行只預置一次'],
  'port.feedback-tx.out': ['',
      'rx 端最新节拍的副本（延迟一轮）',
      'rx 端最新節拍的副本（延遲一輪）'],

  // ---- Shell (control) ----
  'stage.shell.name': ['', 'Shell 命令', 'Shell 指令'],
  'stage.shell.doc': ['',
      '汇聚节点：每收到一个输入节拍（负载不限）就通过 /bin/sh 执行一条 '
      + 'shell 命令；stdout 记为 info，stderr 记为 warn，可选地把 getline '
      + '接到它的标准输入。受会话文件沙盒约束。EOS 结束本阶段。',
      '匯聚節點：每收到一個輸入節拍（負載不限）就透過 /bin/sh 執行一條 '
      + 'shell 指令；stdout 記為 info，stderr 記為 warn，可選地把 getline '
      + '接到它的標準輸入。受工作階段檔案沙箱約束。EOS 結束本階段。'],
  'cfg.shell.command': ['',
      '原样传给 /bin/sh -c 的 shell 命令',
      '原樣傳給 /bin/sh -c 的 shell 指令'],
  'cfg.shell.forward_stdin': ['',
      '把界面的 getline() 接到命令的标准输入（供交互式程序使用）；否则标准'
      + '输入为 /dev/null',
      '把介面的 getline() 接到指令的標準輸入（供互動式程式使用）；否則標準'
      + '輸入為 /dev/null'],
  'cfg.shell.allow': ['',
      '会话处于沙盒中时，该命令唯一被允许 exec 的程序（名称或绝对路径）；留'
      + '空 => 任意程序（无论如何它的写入/网络仍受限）',
      '工作階段處於沙箱中時，該指令唯一被允許 exec 的程式（名稱或絕對路徑）'
      + '；留空 => 任意程式（無論如何它的寫入/網路仍受限）'],
  'cfg.shell.allow_network': ['',
      '会话处于沙盒中时允许访问网络（未启用沙盒时无效）',
      '工作階段處於沙箱中時允許存取網路（未啟用沙箱時無效）'],
  'cfg.shell.timeout_ms': ['',
      '挂钟超时；到期时该命令的进程组会被终止。0 => 不限',
      '掛鐘逾時；到期時該指令的行程群組會被終止。0 => 不限'],
  'port.shell.trigger': ['',
      '任意节拍；每一个都执行一次命令',
      '任意節拍；每一個都執行一次指令'],

  // ---- Video Cleanup (database) ----
  'stage.videos-db-cleanup.name': ['', '视频清理', '影片清理'],
  'stage.videos-db-cleanup.doc': ['',
      '源：周期性清扫 <camera><suffix> LMDB 分段索引，删除早于 '
      + 'retention_seconds 的记录。纯 LMDB 副作用；0 入 / 0 出。',
      '來源：週期性清掃 <camera><suffix> LMDB 分段索引，刪除早於 '
      + 'retention_seconds 的記錄。純 LMDB 副作用；0 入 / 0 出。'],
  'cfg.videos-db-cleanup.camera_name': ['',
      '选定 <camera_name><suffix> 子数据库',
      '選定 <camera_name><suffix> 子資料庫'],
  'cfg.videos-db-cleanup.videos_db_suffix': ['',
      '追加在 camera_name 之后的后缀',
      '附加在 camera_name 之後的後綴'],
  'cfg.videos-db-cleanup.retention_seconds': ['',
      '删除早于此秒数的记录；须大于 0',
      '刪除早於此秒數的記錄；須大於 0'],
  'cfg.videos-db-cleanup.sweep_interval_seconds': ['',
      '两次清扫之间的挂钟秒数；须大于等于 1',
      '兩次清掃之間的掛鐘秒數；須大於等於 1'],
  'cfg.videos-db-cleanup.run_once': ['',
      '清扫一次后即结束',
      '清掃一次後即結束'],

  // ---- Audio VAE Decode (generative) ----
  'stage.audio-vae-decode.name': ['', '音频 VAE 解码', '音訊 VAE 解碼'],
  'stage.audio-vae-decode.doc': ['',
      '在 metal-compute 后端把潜在音轨解码为 PCM。它是 vae-decode 的音频对'
      + '应物，也是 generate-video 音频输出端口的下游。',
      '在 metal-compute 後端把潛在音軌解碼為 PCM。它是 vae-decode 的音訊對'
      + '應物，也是 generate-video 音訊輸出埠的下游。'],
  'cfg.audio-vae-decode.hf_dir': ['',
      '含音频 VAE 的模型目录（从 <hf_dir>/audio_vae 读取）。可选：model 输'
      + '入端口上的 model-select 源会覆盖它',
      '含音訊 VAE 的模型目錄（從 <hf_dir>/audio_vae 讀取）。可選：model 輸'
      + '入埠上的 model-select 來源會覆寫它'],
  'cfg.audio-vae-decode.unload_when_idle': ['',
      '每个节拍处理完就释放 VAE 权重，下一个节拍再重新加载。"auto"（默认）'
      + '根据物理内存与流水线的权重字节数自行判断；"always" / "never" 可强'
      + '制',
      '每個節拍處理完就釋放 VAE 權重，下一個節拍再重新載入。"auto"（預設）'
      + '根據實體記憶體與管線的權重位元組數自行判斷；"always" / "never" 可'
      + '強制'],
  'port.audio-vae-decode.latent': ['',
      'f32 [立体声, 潜通道数, 帧数] 的潜在音频（已白化）',
      'f32 [立體聲, 潛通道數, 影格數] 的潛在音訊（已白化）'],
  'port.audio-vae-decode.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.audio-vae-decode.audio': ['',
      '解码出的 PCM，f32 TensorBeat [声道数, 采样数]，平面排列',
      '解碼出的 PCM，f32 TensorBeat [聲道數, 取樣數]，平面排列'],

  // ---- Audio VAE Encode (generative) ----
  'stage.audio-vae-encode.name': ['', '音频 VAE 编码', '音訊 VAE 編碼'],
  'stage.audio-vae-encode.doc': ['',
      '把 PCM 编码成潜在音轨——即全模态生成器所依据的参考。它是 vae-encode '
      + '的音频对应物，也是 generate-video 的 ref_audio_rows 的上游。它会累'
      + '积所有节拍并在 EOS 时一次性编码：音频 VAE 是因果的且会在时间上压缩'
      + '，因此分块单独编码后无法拼接。',
      '把 PCM 編碼成潛在音軌——即全模態生成器所依據的參考。它是 vae-encode '
      + '的音訊對應物，也是 generate-video 的 ref_audio_rows 的上游。它會累'
      + '積所有節拍並在 EOS 時一次性編碼：音訊 VAE 是因果的且會在時間上壓縮'
      + '，因此分塊單獨編碼後無法拼接。'],
  'cfg.audio-vae-encode.hf_dir': ['',
      '含音频 VAE 的模型目录。可选：model 输入端口上的 model-select 源会覆'
      + '盖它',
      '含音訊 VAE 的模型目錄。可選：model 輸入埠上的 model-select 來源會覆'
      + '寫它'],
  'cfg.audio-vae-encode.sample_rate': ['',
      '后备采样率，仅用于边带未标明采样率的节拍。会标注自身采样率的生产者（'
      + 'audio-to-pcm 就会）让这一项变得不必要',
      '後備取樣率，僅用於邊帶未標明取樣率的節拍。會標註自身取樣率的生產者（'
      + 'audio-to-pcm 就會）讓這一項變得不必要'],
  'cfg.audio-vae-encode.max_seconds': ['',
      '拒绝缓冲超过这么多秒。参考音轨是一段素材而不是实时流，对着一支开着的'
      + '麦克风做无上限缓冲，最终会以 OOM 而不是一条消息收场',
      '拒絕緩衝超過這麼多秒。參考音軌是一段素材而不是即時串流，對著一支開著'
      + '的麥克風做無上限緩衝，最終會以 OOM 而不是一條訊息收場'],
  'cfg.audio-vae-encode.unload_when_idle': ['',
      '编码完成后释放 VAE 权重。"auto"（默认）根据物理内存与流水线的权重字'
      + '节数自行判断；"always" / "never" 可强制',
      '編碼完成後釋放 VAE 權重。"auto"（預設）根據實體記憶體與管線的權重位'
      + '元組數自行判斷；"always" / "never" 可強制'],
  'port.audio-vae-encode.audio': ['',
      'f32 PCM，秩为 1 的 [N] 单声道或秩为 2 的 [声道数, N] 平面排列，取值'
      + '在 [-1, 1]；采样率由 sideband.sample_rate 给出。所有节拍都会被累积'
      + '，编码在 EOS 时只做一次',
      'f32 PCM，秩為 1 的 [N] 單聲道或秩為 2 的 [聲道數, N] 平面排列，取值'
      + '在 [-1, 1]；取樣率由 sideband.sample_rate 給出。所有節拍都會被累積'
      + '，編碼在 EOS 時只做一次'],
  'port.audio-vae-encode.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.audio-vae-encode.latent': ['',
      '编码后的音轨，f32 TensorBeat，采用编码器自身的潜变量形状——对接 '
      + 'generate-video 的 ref_audio_rows 的模型族即 [rows, dim]。每次运行'
      + '在排空时发出一个节拍',
      '編碼後的音軌，f32 TensorBeat，採用編碼器自身的潛變數形狀——對接 '
      + 'generate-video 的 ref_audio_rows 的模型族即 [rows, dim]。每次執行'
      + '在排空時發出一個節拍'],

  // ---- Diffusion Conditioner (generative) ----
  'stage.diffusion-conditioner.name': ['', '扩散条件生成', '擴散條件產生'],
  'stage.diffusion-conditioner.doc': ['',
      '提示词（外加可选的参考图像）-> 扩散 DiT 所需的条件嵌入。本阶段拥有分'
      + '词器 + 文本编码器 +（对能看图的模型）Qwen2.5-VL 视觉塔。它是 '
      + 'generate-image 拆分出来的编码器一半；请与同一个 hf_dir 上的 '
      + 'generate-image 阶段配对使用。在 Mage-Flow 系列模型上，每条提示词（'
      + '编辑任务还包括源图像）都会先经过模型自带的内容政策分类器筛查——这是'
      + '强制的，没有配置项；被拒绝的提示词会得到一张空白图像而不是生成结果'
      + '。',
      '提示詞（外加可選的參考影像）-> 擴散 DiT 所需的條件嵌入。本階段擁有斷'
      + '詞器 + 文字編碼器 +（對能看圖的模型）Qwen2.5-VL 視覺塔。它是 '
      + 'generate-image 拆分出來的編碼器一半；請與同一個 hf_dir 上的 '
      + 'generate-image 階段配對使用。在 Mage-Flow 系列模型上，每條提示詞（'
      + '編輯任務還包括來源影像）都會先經過模型自帶的內容政策分類器篩查——這'
      + '是強制的，沒有設定項；被拒絕的提示詞會得到一張空白影像而不是生成結'
      + '果。'],
  'cfg.diffusion-conditioner.hf_dir': ['',
      '模型目录（text_encoder/、transformer/、tokenizer/）；由 transformer '
      + '的 _class_name 决定模型族与编码器。可选：model 输入端口上的 '
      + 'model-select 源会覆盖它',
      '模型目錄（text_encoder/、transformer/、tokenizer/）；由 transformer '
      + '的 _class_name 決定模型族與編碼器。可選：model 輸入埠上的 '
      + 'model-select 來源會覆寫它'],
  'cfg.diffusion-conditioner.grounded_negative': ['',
      '仅限能看图的模型族：始终在 1 号输出端口发出负向条件——对（可能为空的'
      + '）负向提示词做一次接地编码——好让 DiT 能跑无分类器引导（CFG>1）。与'
      + ' Krea-2 的编辑删除配方一致（空的接地负向条件）。默认 false（只在接'
      + '了非空负向提示词时才发负向条件）',
      '僅限能看圖的模型族：始終在 1 號輸出埠發出負向條件——對（可能為空的）'
      + '負向提示詞做一次接地編碼——好讓 DiT 能跑無分類器引導（CFG>1）。與 '
      + 'Krea-2 的編輯刪除配方一致（空的接地負向條件）。預設 false（只在接'
      + '了非空負向提示詞時才發負向條件）'],
  'cfg.diffusion-conditioner.unload_when_idle': ['',
      '每发出一次条件就释放文本编码器（以及视觉塔），下一条提示词再重新加载'
      + '。编码器在整个去噪过程中都空闲，因此在内存吃紧的机器上，正是它让一'
      + '个大 DiT 与一个大编码器共处一机（4 位的 Boogu-Image 是约 5.6 GB 的'
      + ' DiT，旁边还有约 4.7 GB 的 Qwen3-VL 多模态语言模型）。"auto"（默认'
      + '）综合物理内存、是否有同伴阶段选择了流式加载，以及流水线的权重字节'
      + '数来判断。也可以强制："destroy"（释放字节，代价是每条提示词都要重'
      + '新加载）、"park"（把它们交给内核标为可丢弃——只有机器真的缺内存时才'
      + '回收，否则无需重新加载即可复用），或 "keep"（钉住不放）。旧写法："'
      + 'always" = destroy，"never" = keep',
      '每發出一次條件就釋放文字編碼器（以及視覺塔），下一條提示詞再重新載入'
      + '。編碼器在整個去噪過程中都閒置，因此在記憶體吃緊的機器上，正是它讓'
      + '一個大 DiT 與一個大編碼器共處一機（4 位元的 Boogu-Image 是約 5.6 '
      + 'GB 的 DiT，旁邊還有約 4.7 GB 的 Qwen3-VL 多模態語言模型）。"auto"'
      + '（預設）綜合實體記憶體、是否有同伴階段選擇了串流載入，以及管線的權'
      + '重位元組數來判斷。也可以強制："destroy"（釋放位元組，代價是每條提'
      + '示詞都要重新載入）、"park"（把它們交給核心標為可丟棄——只有機器真的'
      + '缺記憶體時才回收，否則無需重新載入即可重用），或 "keep"（釘住不放'
      + '）。舊寫法："always" = destroy，"never" = keep'],
  'port.diffusion-conditioner.prompt': ['',
      '提示词文本（FlexData 字符串或 {text: ...}）',
      '提示詞文字（FlexData 字串或 {text: ...}）'],
  'port.diffusion-conditioner.negative': ['',
      '可选的负向提示词（FlexData），供 DiT 做无分类器引导；其条件从 1 号输'
      + '出端口发出',
      '可選的負向提示詞（FlexData），供 DiT 做無分類器引導；其條件從 1 號輸'
      + '出埠發出'],
  'port.diffusion-conditioner.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.diffusion-conditioner.ref_image': ['',
      '可选的原始参考图像（平面 U8 RGB TensorBeat [3,H,W]，即 load-image 的'
      + '格式）。能看图的模型族（Qwen-Image-Edit）会把它送进 Qwen2.5-VL 视'
      + '觉塔；其余模型族会忽略它。',
      '可選的原始參考影像（平面 U8 RGB TensorBeat [3,H,W]，即 load-image 的'
      + '格式）。能看圖的模型族（Qwen-Image-Edit）會把它送進 Qwen2.5-VL 視'
      + '覺塔；其餘模型族會忽略它。'],
  'port.diffusion-conditioner.ref_image2': ['',
      '可选的第二张参考图像（格式相同）。Qwen-Image-Edit-2511 支持多参考，'
      + 'Mage-Flow-Edit 的模板也是按参考逐段展开的，因此在这些模型族上视觉'
      + '语言模型必须看到两张图（DiT 的 ref_latent1 只携带第二张的空间细节'
      + '）。每个参考各有自己的视觉块、mROPE 频带和 deepstack 运行。Krea-2 '
      + '的设计只支持单参考，会忽略它。',
      '可選的第二張參考影像（格式相同）。Qwen-Image-Edit-2511 支援多參考，'
      + 'Mage-Flow-Edit 的範本也是按參考逐段展開的，因此在這些模型族上視覺'
      + '語言模型必須看到兩張圖（DiT 的 ref_latent1 只攜帶第二張的空間細節'
      + '）。每個參考各有自己的視覺區塊、mROPE 頻帶和 deepstack 執行。'
      + 'Krea-2 的設計只支援單參考，會忽略它。'],
  'port.diffusion-conditioner.model_config': ['',
      '可选的模型专属参数，来自常驻模型族自己的配置源（krea2-model-config、'
      + 'mage-flow-model-config、boogu-image-model-config、'
      + 'qwen-image-edit-model-config）。目前指的是参考图像如何为接地编码做'
      + '准备：各模型族的参考管道对它的限制各不相同，数值之间不可互换。未接'
      + '时按该模型族自己的数值执行',
      '可選的模型專屬參數，來自常駐模型族自己的組態來源（krea2-model-config'
      + '、mage-flow-model-config、boogu-image-model-config、'
      + 'qwen-image-edit-model-config）。目前指的是參考影像如何為接地編碼做'
      + '準備：各模型族的參考管道對它的限制各不相同，數值之間不可互換。未接'
      + '時按該模型族自己的數值執行'],
  'port.diffusion-conditioner.conditioning': ['',
      '供 generate-image DiT 使用的条件张量（形状与类型随模型族而定：krea2 '
      + 'f16 [n,12,2560]；flux2 f16 [n,3*enc_hidden]；qwen-image-edit 看图'
      + '模式 bf16 [n_real,3584]；mage-flow 看图模式 bf16 [n_real,2560]）',
      '供 generate-image DiT 使用的條件張量（形狀與型別隨模型族而定：krea2 '
      + 'f16 [n,12,2560]；flux2 f16 [n,3*enc_hidden]；qwen-image-edit 看圖'
      + '模式 bf16 [n_real,3584]；mage-flow 看圖模式 bf16 [n_real,2560]）'],
  'port.diffusion-conditioner.neg_conditioning': ['',
      '负向提示词的条件（形状/类型相同）；仅在接了负向提示词时才发出',
      '負向提示詞的條件（形狀/型別相同）；僅在接了負向提示詞時才發出'],

  // ---- Diffusion Sampler (generative) ----
  'stage.diffusion-sampler-select.name': ['', '扩散采样器', '擴散取樣器'],
  'stage.diffusion-sampler-select.doc': ['',
      '选择扩散采样器/积分器（euler | heun | dpmpp_2m | dpmpp_sde | dmd），'
      + '把它的配置作为 FlexData 节拍发出，供 generate-image 阶段锁存。与 '
      + 'scheduler-select 配套。语言模型的 token 采样器请改用 '
      + 'sampler-select。0 入 / 1 出（只发一次）。',
      '選擇擴散取樣器/積分器（euler | heun | dpmpp_2m | dpmpp_sde | dmd），'
      + '把它的組態作為 FlexData 節拍發出，供 generate-image 階段鎖存。與 '
      + 'scheduler-select 配套。語言模型的 token 取樣器請改用 '
      + 'sampler-select。0 入 / 1 出（只發一次）。'],
  'cfg.diffusion-sampler-select.method': ['',
      '采样方法：euler（默认）| heun | dpmpp_2m | dpmpp_sde | dmd（'
      + 'Boogu-Image Turbo 的少步学生模型：先跳到 x0 再重新加噪；只对 DMD '
      + '蒸馏过的检查点有意义）',
      '取樣方法：euler（預設）| heun | dpmpp_2m | dpmpp_sde | dmd（'
      + 'Boogu-Image Turbo 的少步學生模型：先跳到 x0 再重新加噪；只對 DMD '
      + '蒸餾過的檢查點有意義）'],
  'cfg.diffusion-sampler-select.eta': ['',
      'dpmpp_sde 的随机性强度，0 = 确定性（默认 1.0）',
      'dpmpp_sde 的隨機性強度，0 = 確定性（預設 1.0）'],
  'cfg.diffusion-sampler-select.s_noise': ['',
      'dpmpp_sde 附加噪声的比例（默认 1.0）',
      'dpmpp_sde 附加雜訊的比例（預設 1.0）'],
  'cfg.diffusion-sampler-select.seed': ['',
      'dpmpp_sde / dmd 重新加噪的随机种子（默认 0）',
      'dpmpp_sde / dmd 重新加噪的隨機種子（預設 0）'],
  'cfg.diffusion-sampler-select.conditioning_sigma': ['',
      '仅 dmd：噪声计划的起始 sigma，linspace(该值, 1, steps+1)[:-1]（默认 '
      + '0.0——参考实现的图像编辑设定；文生图脚本传 0.001）',
      '僅 dmd：雜訊排程的起始 sigma，linspace(該值, 1, steps+1)[:-1]（預設 '
      + '0.0——參考實作的影像編輯設定；文生圖指令碼傳 0.001）'],
  'port.diffusion-sampler-select.sampler': ['',
      '采样器配置 {sampler,method,eta,s_noise,seed,+conditioning_sigma}',
      '取樣器組態 {sampler,method,eta,s_noise,seed,+conditioning_sigma}'],

  // ---- Generate Image (generative) ----
  'stage.generate-image.name': ['', '生成图片', '生成影像'],
  'stage.generate-image.doc': ['',
      '扩散 DiT 去噪器：条件（来自 diffusion-conditioner 阶段）-> 模型族的 '
      + 'MMDiT -> FlowMatchEuler -> 潜变量，运行在 metal-compute 后端。这是'
      + '拆分出来的去噪器一半（下游接 vae-decode）。',
      '擴散 DiT 去噪器：條件（來自 diffusion-conditioner 階段）-> 模型族的 '
      + 'MMDiT -> FlowMatchEuler -> 潛變數，執行在 metal-compute 後端。這是'
      + '拆分出來的去噪器一半（下游接 vae-decode）。'],
  'cfg.generate-image.hf_dir': ['',
      'Krea-2-Turbo / FLUX.2 模型目录（text_encoder/、transformer/、'
      + 'tokenizer/）；可以是原版管道，也可以是经 model-quantize 处理过的自'
      + '包含管道。可选：model 输入端口上的 model-select 源会覆盖它',
      'Krea-2-Turbo / FLUX.2 模型目錄（text_encoder/、transformer/、'
      + 'tokenizer/）；可以是原版管道，也可以是經 model-quantize 處理過的自'
      + '包含管道。可選：model 輸入埠上的 model-select 來源會覆寫它'],
  'cfg.generate-image.dit_dir': ['',
      '覆盖 DiT 目录（例如一个量化过的 4/8 位 DiT）；否则取 <hf_dir>/'
      + 'transformer',
      '覆寫 DiT 目錄（例如一個量化過的 4/8 位元 DiT）；否則取 <hf_dir>/'
      + 'transformer'],
  'cfg.generate-image.strength': ['',
      '图生图强度，取值 [0,1]；0（默认）表示从噪声开始的文生图（初始潜变量'
      + '由 vae-encode 从 `latent` 输入端口送入）',
      '圖生圖強度，取值 [0,1]；0（預設）表示從雜訊開始的文生圖（初始潛變數'
      + '由 vae-encode 從 `latent` 輸入埠送入）'],
  'cfg.generate-image.height': ['',
      '输出高度，须为 16 的倍数。与 width 一同留空 = 按该模型族的 VAE 缩放'
      + '比例从 ref_latent0（5 号输入端口）推断；若也没有参考图，则为 256',
      '輸出高度，須為 16 的倍數。與 width 一同留空 = 按該模型族的 VAE 縮放'
      + '比例從 ref_latent0（5 號輸入埠）推斷；若也沒有參考影像，則為 256'],
  'cfg.generate-image.width': ['',
      '输出宽度，须为 16 的倍数。与 height 一同留空 = 从 ref_latent0（5 号'
      + '输入端口）推断；若也没有参考图，则为 256',
      '輸出寬度，須為 16 的倍數。與 height 一同留空 = 從 ref_latent0（5 號'
      + '輸入埠）推斷；若也沒有參考影像，則為 256'],
  'cfg.generate-image.steps': ['',
      'turbo 采样器步数（默认 8）',
      'turbo 取樣器步數（預設 8）'],
  'cfg.generate-image.seed': ['',
      '初始噪声的随机种子（默认 0）',
      '初始雜訊的隨機種子（預設 0）'],
  'cfg.generate-image.guidance_scale': ['',
      '无分类器引导强度；1（默认）表示关闭 CFG。大于 1 且 1 号输入端口接了'
      + '负向提示词时，每一步会多跑一次 DiT 前向',
      '無分類器引導強度；1（預設）表示關閉 CFG。大於 1 且 1 號輸入埠接了負'
      + '向提示詞時，每一步會多跑一次 DiT 前向'],
  'cfg.generate-image.init_latents': ['',
      '调试用：原始 f32 打包初始潜变量 [img_seq, 64]（用于复现/黄金样本）',
      '偵錯用：原始 f32 打包初始潛變數 [img_seq, 64]（用於重現/黃金樣本）'],
  'cfg.generate-image.i8_gemm': ['',
      '加速模式（有损）：DiT 大块矩阵乘改用动态 int8 GEMM，速率约为 f16 的 '
      + '2 倍，精度即 int8 水平；没有 NAX matmul2d（矩阵核心 GPU + 相应内核'
      + '）时会被忽略。默认 false；环境变量 VPIPE_I8_GEMM 可覆盖',
      '加速模式（失真）：DiT 大塊矩陣乘改用動態 int8 GEMM，速率約為 f16 的 '
      + '2 倍，精度即 int8 水準；沒有 NAX matmul2d（矩陣核心 GPU + 相應核心'
      + '）時會被忽略。預設 false；環境變數 VPIPE_I8_GEMM 可覆寫'],
  'cfg.generate-image.lora': ['',
      '在运行时生效的 LoRA——每个被适配的投影计算 W x + scale * B (A x)，而'
      + '不是把增量折进权重里；对 4 位基座而言，这就是“保住一个小修正”和“把'
      + '它四舍五入抹掉”的区别。可以是一个已注册的模型、一个含单个 '
      + '.safetensors 的目录，或指向该文件的路径。仅 Krea-2 与 FLUX.2 支持'
      + '。加载时生效：它决定各 block 如何构建，因此 DiT 起来之后再改的节拍'
      + '只会被报告并忽略。模型族 model-config 节拍上的 `lora` 会覆盖这里的'
      + '设置',
      '在執行階段生效的 LoRA——每個被套用的投影計算 W x + scale * B (A x)，'
      + '而不是把增量摺進權重裡；對 4 位元基座而言，這就是「保住一個小修正'
      + '」和「把它四捨五入抹掉」的差別。可以是一個已註冊的模型、一個含單個'
      + ' .safetensors 的目錄，或指向該檔案的路徑。僅 Krea-2 與 FLUX.2 支援'
      + '。載入時生效：它決定各 block 如何建構，因此 DiT 起來之後再改的節拍'
      + '只會被報告並忽略。模型族 model-config 節拍上的 `lora` 會覆寫這裡的'
      + '設定'],
  'cfg.generate-image.lora_scale': ['',
      '适配器强度，按每次前向生效。可实时调整：它作为常数随 GEMM 一起计算，'
      + '因此无需重新加载即可扫参。1.0 = 与训练时一致；0 会跳过两个适配器 '
      + 'GEMM，所以关掉就是真的关掉',
      '轉接器強度，按每次前向生效。可即時調整：它作為常數隨 GEMM 一起計算，'
      + '因此無需重新載入即可掃參。1.0 = 與訓練時一致；0 會跳過兩個轉接器 '
      + 'GEMM，所以關掉就是真的關掉'],
  'port.generate-image.conditioning': ['',
      '来自 diffusion-conditioner 阶段的条件张量（形状与类型随模型族而定）',
      '來自 diffusion-conditioner 階段的條件張量（形狀與型別隨模型族而定）'],
  'port.generate-image.neg_conditioning': ['',
      '可选的负向条件（条件生成阶段的 1 号输出端口），用于无分类器引导',
      '可選的負向條件（條件產生階段的 1 號輸出埠），用於無分類器引導'],
  'port.generate-image.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.generate-image.sampler': ['',
      '可选的采样器配置 FlexData（diffusion-sampler-select）',
      '可選的取樣器組態 FlexData（diffusion-sampler-select）'],
  'port.generate-image.scheduler': ['',
      '可选的噪声计划配置 FlexData（scheduler-select）',
      '可選的雜訊排程組態 FlexData（scheduler-select）'],
  'port.generate-image.ref_latent0': ['',
      '可选的参考潜变量 0（来自 vae-encode 的通道优先 f32）；用于 FLUX.2/'
      + 'QIE 的条件，或 Krea-2 的图生图初值',
      '可選的參考潛變數 0（來自 vae-encode 的通道優先 f32）；用於 FLUX.2/'
      + 'QIE 的條件，或 Krea-2 的圖生圖初值'],
  'port.generate-image.ref_latent1': ['',
      '可选的参考潜变量 1（来自 vae-encode 的通道优先 f32）；FLUX.2/QIE 的'
      + '第二个参考（RoPE 位置不同）；Krea-2 会忽略它',
      '可選的參考潛變數 1（來自 vae-encode 的通道優先 f32）；FLUX.2/QIE 的'
      + '第二個參考（RoPE 位置不同）；Krea-2 會忽略它'],
  'port.generate-image.model_config': ['',
      '可选的模型专属参数，来自常驻模型族自己的配置源——flux2-model-config（'
      + 'klein-kv 配方及其 `lora`）、krea2-model-config（它的 `lora`），或 '
      + 'mage-flow-model-config（来源水印）。这些参数原样传给该族自己的参数'
      + '结构，本阶段不解读，因此这里不设任何该类旋钮。接上它会把 DiT 的加'
      + '载推迟到第一个节拍，因为像 klein_kv 这样的配方——或者决定各 block '
      + '如何构建的适配器——必须在读取权重之前就确定。适配器相关的键只在这里'
      + '读取，别无他处：只接到 diffusion-conditioner 的 krea2-model-config'
      + ' 会送达它的 `vl_*` 各键，而它的 `lora` 则悄无声息地无处生效',
      '可選的模型專屬參數，來自常駐模型族自己的組態來源——flux2-model-config'
      + '（klein-kv 配方及其 `lora`）、krea2-model-config（它的 `lora`），'
      + '或 mage-flow-model-config（來源浮水印）。這些參數原樣傳給該族自己'
      + '的參數結構，本階段不解讀，因此這裡不設任何該類旋鈕。接上它會把 DiT'
      + ' 的載入延後到第一個節拍，因為像 klein_kv 這樣的配方——或者決定各 '
      + 'block 如何建構的轉接器——必須在讀取權重之前就確定。轉接器相關的鍵只'
      + '在這裡讀取，別無他處：只接到 diffusion-conditioner 的 '
      + 'krea2-model-config 會送達它的 `vl_*` 各鍵，而它的 `lora` 則悄無聲'
      + '息地無處生效'],
  'port.generate-image.latent': ['',
      'f32 潜变量 [z_dim, H/8, W/8]（未打包，已白化）',
      'f32 潛變數 [z_dim, H/8, W/8]（未打包，已白化）'],
  'port.generate-image.step_latent': ['',
      '可选的逐步潜变量（每个采样步一个节拍，格式与 `latent` 相同）——在这里'
      + '接一个 vae-decode 即可把去噪过程可视化，便于调试。只有接上时才发出'
      + '。',
      '可選的逐步潛變數（每個取樣步一個節拍，格式與 `latent` 相同）——在這裡'
      + '接一個 vae-decode 即可把去噪過程視覺化，便於偵錯。只有接上時才發出'
      + '。'],

  // ---- Generate Video (generative) ----
  'stage.generate-video.name': ['', '生成视频', '生成影片'],
  'stage.generate-video.doc': ['',
      '视频 DiT 去噪器：条件（外加可选的关键帧或参考潜变量）-> 常驻模型族在'
      + '其 transformer 上的采样器 -> 一段潜在视频；对会生成配乐的模型族，'
      + '还会输出一段潜在音轨。运行在 metal-compute 后端。下游接 vae-decode'
      + '（以及 audio-vae-decode）。',
      '影片 DiT 去噪器：條件（外加可選的關鍵影格或參考潛變數）-> 常駐模型族'
      + '在其 transformer 上的取樣器 -> 一段潛在影片；對會產生配樂的模型族'
      + '，還會輸出一段潛在音軌。執行在 metal-compute 後端。下游接 '
      + 'vae-decode（以及 audio-vae-decode）。'],
  'cfg.generate-video.hf_dir': ['',
      '视频模型根目录。目录结构由常驻模型族自己决定——可能是上游那套按组件分'
      + '目录的树，也可能是命名不同的重打包版本——每个已注册的模型族会认领它'
      + '能识别的根目录，因此这里只要给出检查点根目录，由认领它的模型族决定'
      + '其余细节。可选：model 输入端口上的 model-select 源会覆盖它',
      '影片模型根目錄。目錄結構由常駐模型族自己決定——可能是上游那套按元件分'
      + '目錄的樹，也可能是命名不同的重新打包版本——每個已註冊的模型族會認領'
      + '它能辨識的根目錄，因此這裡只要給出檢查點根目錄，由認領它的模型族決'
      + '定其餘細節。可選：model 輸入埠上的 model-select 來源會覆寫它'],
  'cfg.generate-video.height': ['',
      '视频高度（像素）。会向上取整到常驻模型族的 VAE 步幅乘以其 DiT patch '
      + '的倍数，各族不同，因此任何正值都接受，阶段会记录实际使用的值',
      '影片高度（像素）。會向上取整到常駐模型族的 VAE 步幅乘以其 DiT patch '
      + '的倍數，各族不同，因此任何正值都接受，階段會記錄實際使用的值'],
  'cfg.generate-video.width': ['',
      '视频宽度（像素）；倍数规则与高度相同',
      '影片寬度（像素）；倍數規則與高度相同'],
  'cfg.generate-video.frames': ['',
      '视频帧数。会向上取整到常驻模型 VAE 能分块处理的最近帧数，各族不同，'
      + '因此这里接受任何正数',
      '影片影格數。會向上取整到常駐模型 VAE 能分塊處理的最近影格數，各族不'
      + '同，因此這裡接受任何正數'],
  'cfg.generate-video.fps': ['',
      '打在输出潜变量上的帧率，供下游的解码器与编码器使用。本阶段不做重采样'
      + '：该值描述的是模型族生成出来的片段，因此以其他帧率训练的模型族需要'
      + '显式设置它',
      '打在輸出潛變數上的影格率，供下游的解碼器與編碼器使用。本階段不做重新'
      + '取樣：該值描述的是模型族產生出來的片段，因此以其他影格率訓練的模型'
      + '族需要明確設定它'],
  'cfg.generate-video.steps': ['', '去噪步数', '去噪步數'],
  'cfg.generate-video.seed': ['',
      '初始噪声的随机种子（默认 0）',
      '初始雜訊的隨機種子（預設 0）'],
  'cfg.generate-video.i8_gemm': ['',
      '加速模式（有损）：DiT 大块矩阵乘改用动态 int8 GEMM 代替 bf16，精度即'
      + ' int8 水平。只有前向会为矩阵核心物化稠密权重的模型族才会采用；纯 '
      + 'steel 的模型族会忽略它，没有 NAX matmul2d 的机器同样忽略。它还会按'
      + '行数自门控（>= 1024），因此短片段无论如何都保持 bf16 分块；也会按'
      + '投影为凑齐一个完整 int8 分块需要补多少零来判断（额外 MAC 超过 10% '
      + '即拒绝）。实测：minimax-h3，生产分辨率下的 2 个 block，2.02-2.11 '
      + '秒/步 -> 1.52-1.55 秒（1.35 倍），该步的速度场 rms 变化 0.11%。默'
      + '认 false；环境变量 VPIPE_I8_GEMM 可覆盖',
      '加速模式（失真）：DiT 大塊矩陣乘改用動態 int8 GEMM 取代 bf16，精度即'
      + ' int8 水準。只有前向會為矩陣核心具現稠密權重的模型族才會採用；純 '
      + 'steel 的模型族會忽略它，沒有 NAX matmul2d 的機器同樣忽略。它還會按'
      + '列數自我把關（>= 1024），因此短片段無論如何都保持 bf16 分塊；也會'
      + '按投影為湊齊一個完整 int8 分塊需要補多少零來判斷（額外 MAC 超過 10'
      + '% 即拒絕）。實測：minimax-h3，生產解析度下的 2 個 block，2.02-2.11'
      + ' 秒/步 -> 1.52-1.55 秒（1.35 倍），該步的速度場 rms 變化 0.11%。預'
      + '設 false；環境變數 VPIPE_I8_GEMM 可覆寫'],
  'cfg.generate-video.unload_when_idle': ['',
      '每生成完一个片段就释放常驻模型的权重，下一个片段再重新加载。"auto"（'
      + '默认）根据物理内存与流水线权重字节数自行判断；"always" / "never" '
      + '可强制',
      '每產生完一個片段就釋放常駐模型的權重，下一個片段再重新載入。"auto"（'
      + '預設）根據實體記憶體與管線權重位元組數自行判斷；"always" / "never"'
      + ' 可強制'],
  'port.generate-video.conditioning': ['',
      '来自 diffusion-conditioner 的文本隐状态：bf16 [text_seq, dim]，宽度'
      + '即常驻模型族自身编码器的宽度。条件生成阶段解析的是同一个检查点，因'
      + '此宽度是构造上就一致的，不需要配置',
      '來自 diffusion-conditioner 的文字隱狀態：bf16 [text_seq, dim]，寬度'
      + '即常駐模型族自身編碼器的寬度。條件產生階段解析的是同一個檢查點，因'
      + '此寬度是結構上就一致的，不需要設定'],
  'port.generate-video.neg_conditioning': ['',
      '可选的负向条件（条件生成阶段的 1 号输出端口），用于无分类器引导。未'
      + '经引导蒸馏的模型族需要它：没有它，引导强度会被强制为 1。经过引导蒸'
      + '馏的模型族每步只做一次前向，会忽略它',
      '可選的負向條件（條件產生階段的 1 號輸出埠），用於無分類器引導。未經'
      + '引導蒸餾的模型族需要它：沒有它，引導強度會被強制為 1。經過引導蒸餾'
      + '的模型族每步只做一次前向，會忽略它'],
  'port.generate-video.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.generate-video.sampler': ['',
      '可选的采样器配置 FlexData（diffusion-sampler-select）。由“按配置采样'
      + '”的模型族读取，用来替换它自带的采样器；对自带采样器和自带噪声计划'
      + '的模型族无效——那类模型族改用它自己的 model-config 源来调整',
      '可選的取樣器組態 FlexData（diffusion-sampler-select）。由「按組態取'
      + '樣」的模型族讀取，用來取代它自帶的取樣器；對自帶取樣器和自帶雜訊排'
      + '程的模型族無效——那類模型族改用它自己的 model-config 來源來調整'],
  'port.generate-video.scheduler': ['',
      '可选的噪声计划配置 FlexData（scheduler-select）。对 sampler 端口无效'
      + '的那些模型族同样无效',
      '可選的雜訊排程組態 FlexData（scheduler-select）。對 sampler 埠無效的'
      + '那些模型族同樣無效'],
  'port.generate-video.ref_latent0': ['',
      '可选的图生视频条件潜变量，来自 vae-encode，采用常驻模型族自身的潜空'
      + '间几何。具体形状取决于该族如何施加条件：可能是片段形状的张量（条件'
      + '图像后接空白帧），也可能是单张首帧关键帧锚点。接上它即表示图生视频',
      '可選的圖生影片條件潛變數，來自 vae-encode，採用常駐模型族自身的潛空'
      + '間幾何。具體形狀取決於該族如何施加條件：可能是片段形狀的張量（條件'
      + '影像後接空白影格），也可能是單張首格關鍵影格錨點。接上它即表示圖生'
      + '影片'],
  'port.generate-video.ref_latent1': ['',
      '可选的末帧关键帧锚点，来自第二个 vae-encode，供以关键帧（而非单个片'
      + '段形状潜变量）作锚点的模型族使用。与 ref_latent0 一起构成首尾帧模'
      + '式',
      '可選的末格關鍵影格錨點，來自第二個 vae-encode，供以關鍵影格（而非單'
      + '個片段形狀潛變數）作錨點的模型族使用。與 ref_latent0 一起構成首尾'
      + '格模式'],
  'port.generate-video.ref_video_rows': ['',
      '可选的参考视频行，来自 video-ref-encoder：f32 [rows, 96]，供带参考条'
      + '件分区的模型族使用。它们已经打包成 DiT 的行并按参考顺序拼接，因为'
      + '每个参考都以各自的分辨率编码，行是它们唯一共有的形状。接上它就把本'
      + '阶段切换到参考条件模式；这些行所属的几何信息随条件节拍的边带一起传'
      + '来',
      '可選的參考影片列，來自 video-ref-encoder：f32 [rows, 96]，供帶參考條'
      + '件分區的模型族使用。它們已經打包成 DiT 的列並按參考順序串接，因為'
      + '每個參考都以各自的解析度編碼，列是它們唯一共有的形狀。接上它就把本'
      + '階段切換到參考條件模式；這些列所屬的幾何資訊隨條件節拍的邊帶一起傳'
      + '來'],
  'port.generate-video.ref_audio_rows': ['',
      '可选的参考音频行，来自同一个 video-ref-encoder：f32 [rows, 32]，单个'
      + '参考内部按声道优先排列。它们以干净时间步随行，永远不参与去噪',
      '可選的參考音訊列，來自同一個 video-ref-encoder：f32 [rows, 32]，單個'
      + '參考內部按聲道優先排列。它們以乾淨時間步隨行，永遠不參與去噪'],
  'port.generate-video.model_config': ['',
      '可选的模型专属参数，来自常驻模型族自己的配置源——每个模型族都带一个，'
      + '承载只有它才需要说明的内容（引导强度与专家切换边界、sigma shift、'
      + '条件时间步、音频时长）。这些参数原样传给该族的 GenerationParams，'
      + '本阶段不解读，因此这里不设任何该类旋钮。未接时模型族按其文档默认值'
      + '运行',
      '可選的模型專屬參數，來自常駐模型族自己的組態來源——每個模型族都帶一個'
      + '，承載只有它才需要說明的內容（引導強度與專家切換邊界、sigma shift'
      + '、條件時間步、音訊長度）。這些參數原樣傳給該族的 GenerationParams'
      + '，本階段不解讀，因此這裡不設任何該類旋鈕。未接時模型族按其文件預設'
      + '值執行'],
  'port.generate-video.audio_conditioning': ['',
      '可选的第二模态条件，供那些用自己对提示词的另一路投影来生成配乐的模型'
      + '族使用：与 0 号输入端口上的视频条件并列的一路音频条件，宽度自成一'
      + '套。之所以单列一个端口，是因为它来自不同的投影、宽度也不同，把两者'
      + '塞进同一个节拍会让这种拆分沦为约定而非类型。未接时，这类模型族只生'
      + '成视频并明确说明；它绝不能自行编造一个全零的条件——那等于在说“静音”'
      + '，而不是“没有”。不接受第二路条件的模型族会忽略它',
      '可選的第二模態條件，供那些用自己對提示詞的另一路投影來產生配樂的模型'
      + '族使用：與 0 號輸入埠上的影片條件並列的一路音訊條件，寬度自成一套'
      + '。之所以單列一個埠，是因為它來自不同的投影、寬度也不同，把兩者塞進'
      + '同一個節拍會讓這種拆分淪為約定而非型別。未接時，這類模型族只產生影'
      + '片並明確說明；它絕不能自行編造一個全零的條件——那等於在說「靜音」，'
      + '而不是「沒有」。不接受第二路條件的模型族會忽略它'],
  'port.generate-video.latent': ['',
      'f32 潜在视频 [z, T, H/r, W/r]（已白化），边带 {fps, frames}。z/r 即'
      + '常驻模型族的 VAE 几何参数',
      'f32 潛在影片 [z, T, H/r, W/r]（已白化），邊帶 {fps, frames}。z/r 即'
      + '常駐模型族的 VAE 幾何參數'],
  'port.generate-video.audio_latent': ['',
      'f32 潜在音频 [audio_channels, audio_latents]，来自会生成配乐的模型族'
      + '——不生成配乐的模型族永远不会写它',
      'f32 潛在音訊 [audio_channels, audio_latents]，來自會產生配樂的模型族'
      + '——不產生配樂的模型族永遠不會寫它'],

  // ---- Model Select (generative) ----
  'stage.model-select.name': ['', '模型选择', '模型選擇'],
  'stage.model-select.doc': ['',
      '选定一个扩散模型目录，作为 FlexData 节拍发出，供 '
      + 'diffusion-conditioner / generate-image / generate-video / '
      + 'vae-encode / vae-decode / audio-vae-decode 共用——各阶段的 model 输'
      + '入端口会覆盖其自身的 hf_dir 配置。0 入 / 1 出（只发一次）。',
      '選定一個擴散模型目錄，作為 FlexData 節拍發出，供 '
      + 'diffusion-conditioner / generate-image / generate-video / '
      + 'vae-encode / vae-decode / audio-vae-decode 共用——各階段的 model 輸'
      + '入埠會覆寫其自身的 hf_dir 設定。0 入 / 1 出（只發一次）。'],
  'cfg.model-select.hf_dir': ['',
      '由 diffusion-conditioner、generate-image / generate-video（DiT）、'
      + 'vae-encode、vae-decode 与 audio-vae-decode 共用的模型目录/注册键；'
      + '以节拍形式发出，覆盖它们各自的 hf_dir 配置项',
      '由 diffusion-conditioner、generate-image / generate-video（DiT）、'
      + 'vae-encode、vae-decode 與 audio-vae-decode 共用的模型目錄/註冊鍵；'
      + '以節拍形式發出，覆寫它們各自的 hf_dir 設定項'],
  'port.model-select.model': ['',
      '共享模型引用 { hf_dir }，作为 FlexData 节拍供扩散类阶段的 model 输入'
      + '端口锁存',
      '共享模型參照 { hf_dir }，作為 FlexData 節拍供擴散類階段的 model 輸入'
      + '埠鎖存'],

  // ---- Sampler (generative) ----
  'stage.sampler-select.name': ['', '采样器', '取樣器'],
  'stage.sampler-select.doc': ['',
      '设定语言模型的 token 采样器（temperature / top_k / top_p / min_p / '
      + '各类惩罚 / seed），把配置作为 FlexData 节拍发出，供 text-chat、'
      + 'visual-qa、realtime-vqa、audio-transcribe 或 text-to-speech 阶段锁'
      + '存。所有旋钮都取默认值即等于 argmax，因此至少要设置 temperature 才'
      + '会真正采样。扩散积分器请改用 diffusion-sampler-select。0 入 / 1 出'
      + '（只发一次）。',
      '設定語言模型的 token 取樣器（temperature / top_k / top_p / min_p / '
      + '各類懲罰 / seed），把組態作為 FlexData 節拍發出，供 text-chat、'
      + 'visual-qa、realtime-vqa、audio-transcribe 或 text-to-speech 階段鎖'
      + '存。所有旋鈕都取預設值即等於 argmax，因此至少要設定 temperature 才'
      + '會真正取樣。擴散積分器請改用 diffusion-sampler-select。0 入 / 1 出'
      + '（只發一次）。'],
  'cfg.sampler-select.temperature': ['',
      'softmax 温度；<= 0 时强制取最大值（贪心）',
      'softmax 溫度；<= 0 時強制取最大值（貪婪）'],
  'cfg.sampler-select.top_k': ['',
      '仅保留前 k 个 logits；0 = 禁用',
      '僅保留前 k 個 logits；0 = 停用'],
  'cfg.sampler-select.top_p': ['',
      '核采样：累积概率达到 p 的最短前缀；1.0 = 禁用',
      '核取樣：累積機率達到 p 的最短前綴；1.0 = 停用'],
  'cfg.sampler-select.min_p': ['',
      '丢弃低于 min_p * 最大概率的 token；0 = 禁用',
      '丟棄低於 min_p * 最大機率的 token；0 = 停用'],
  'cfg.sampler-select.repetition_penalty': ['',
      '对已出现的 token 施加惩罚；1.0 = 禁用',
      '對已出現的 token 施加懲罰；1.0 = 停用'],
  'cfg.sampler-select.presence_penalty': ['',
      '对已出现 token 的固定惩罚；0.0 = 禁用',
      '對已出現 token 的固定懲罰；0.0 = 停用'],
  'cfg.sampler-select.seed': ['',
      '采样随机数种子；0 = 每次重新随机',
      '取樣隨機數種子；0 = 每次重新隨機'],
  'port.sampler-select.sampler': ['',
      'token 采样器配置 {sampler:"token",temperature,top_k,top_p,min_p,'
      + 'repetition_penalty,presence_penalty,seed}',
      'token 取樣器組態 {sampler:"token",temperature,top_k,top_p,min_p,'
      + 'repetition_penalty,presence_penalty,seed}'],

  // ---- Scheduler Select (generative) ----
  'stage.scheduler-select.name': ['', '噪声计划表', '雜訊排程器'],
  'stage.scheduler-select.doc': ['',
      '选择扩散的 sigma 计划（simple | karras | exponential | boogu_v1）以'
      + '及步数/shift，把配置作为 FlexData 节拍发出，供 generate-image 阶段'
      + '锁存。与模型无关（原样转发用户的选择）；这里的配置项会覆盖 turbo '
      + '默认值。与 diffusion-sampler-select 配套。0 入 / 1 出（只发一次）'
      + '。',
      '選擇擴散的 sigma 排程（simple | karras | exponential | boogu_v1）以'
      + '及步數/shift，把組態作為 FlexData 節拍發出，供 generate-image 階段'
      + '鎖存。與模型無關（原樣轉發使用者的選擇）；這裡的設定項會覆寫 turbo'
      + ' 預設值。與 diffusion-sampler-select 配套。0 入 / 1 出（只發一次）'
      + '。'],
  'cfg.scheduler-select.type': ['',
      '计划类型：simple（默认）| karras | exponential | boogu_v1（'
      + 'Boogu-Image 的 logistic 时间偏移；它的 sigma 是递增的——0 是噪声，1'
      + ' 是干净——因此只适用于 Boogu 的 DiT）',
      '排程類型：simple（預設）| karras | exponential | boogu_v1（'
      + 'Boogu-Image 的 logistic 時間偏移；它的 sigma 是遞增的——0 是雜訊，1'
      + ' 是乾淨——因此只適用於 Boogu 的 DiT）'],
  'cfg.scheduler-select.steps': ['',
      '覆盖去噪步数（默认 8）',
      '覆寫去噪步數（預設 8）'],
  'cfg.scheduler-select.shift': ['',
      '覆盖 mu / 时间偏移强度（默认 1.15）',
      '覆寫 mu / 時間偏移強度（預設 1.15）'],
  'cfg.scheduler-select.shift_type': ['',
      '时间偏移形式：exponential（默认）| linear',
      '時間偏移形式：exponential（預設）| linear'],
  'cfg.scheduler-select.rho': ['',
      'karras 曲率（默认 7）',
      'karras 曲率（預設 7）'],
  'cfg.scheduler-select.seq_len': ['',
      '仅 boogu_v1：据以读取 mu 的静态 token 数，即检查点 scheduler_config '
      + '中的 seq_len（默认 4096 = 1K 方图）',
      '僅 boogu_v1：據以讀取 mu 的靜態 token 數，即檢查點 scheduler_config '
      + '中的 seq_len（預設 4096 = 1K 方圖）'],
  'cfg.scheduler-select.base_shift': ['',
      '仅 boogu_v1：base_seq 个 token 处的 mu（默认 0.5）',
      '僅 boogu_v1：base_seq 個 token 處的 mu（預設 0.5）'],
  'cfg.scheduler-select.max_shift': ['',
      '仅 boogu_v1：max_seq 个 token 处的 mu（默认 1.15）',
      '僅 boogu_v1：max_seq 個 token 處的 mu（預設 1.15）'],
  'port.scheduler-select.scheduler': ['',
      '噪声计划配置 {scheduler,type,steps,shift,shift_type,rho}（boogu_v1 '
      + '另加 base_shift/max_shift/seq_len）',
      '雜訊排程組態 {scheduler,type,steps,shift,shift_type,rho}（boogu_v1 '
      + '另加 base_shift/max_shift/seq_len）'],

  // ---- Speak (generative) ----
  'stage.text-to-speech.name': ['', '语音合成', '文字轉語音'],
  'stage.text-to-speech.doc': ['',
      '文本转语音（MOSS-TTS，metal）：把每个输入文本节拍合成为 PCM 波形并作'
      + '为 TensorBeat 发出。具体变体由语言模型目录 config.json 中的 '
      + 'model_type 决定："moss_tts"（8B 延迟模式 -> 24 kHz 单声道）或 "'
      + 'moss_tts_local"（v1.5 深度解码器 -> 48 kHz 立体声）。',
      '文字轉語音（MOSS-TTS，metal）：把每個輸入文字節拍合成為 PCM 波形並作'
      + '為 TensorBeat 發出。具體變體由語言模型目錄 config.json 中的 '
      + 'model_type 決定："moss_tts"（8B 延遲模式 -> 24 kHz 單聲道）或 "'
      + 'moss_tts_local"（v1.5 深度解碼器 -> 48 kHz 立體聲）。'],
  'cfg.text-to-speech.hf_dir': ['',
      'MOSS-TTS 语言模型：models 数据库中的键（由 model-fetch / '
      + 'model-quantize 注册）或 HF 风格的模型目录；同名时数据库键优先于路'
      + '径。可选 8B（moss-tts）、v1.5（moss-tts-local）或实时版（'
      + 'moss-tts-realtime）。',
      'MOSS-TTS 語言模型：models 資料庫中的鍵（由 model-fetch / '
      + 'model-quantize 註冊）或 HF 風格的模型目錄；同名時資料庫鍵優先於路'
      + '徑。可選 8B（moss-tts）、v1.5（moss-tts-local）或即時版（'
      + 'moss-tts-realtime）。'],
  'cfg.text-to-speech.codec_dir': ['',
      'MOSS-Audio-Tokenizer（编解码器）模型：models 数据库中的键或文件系统'
      + '路径；同名时数据库键优先于路径。需与语言模型变体匹配：moss-codec（'
      + '8B）或 moss-codec-v2（v1.5）。',
      'MOSS-Audio-Tokenizer（編解碼器）模型：models 資料庫中的鍵或檔案系統'
      + '路徑；同名時資料庫鍵優先於路徑。需與語言模型變體相符：moss-codec（'
      + '8B）或 moss-codec-v2（v1.5）。'],
  'cfg.text-to-speech.codec_quant': ['',
      '编解码器权重精度："int8" 表示把编解码器 transformer 的 GEMM 权重存成'
      + ' int8 分组 32 的仿射量化（常驻占用约减半，音质代价很小）；默认/留'
      + '空 = f16',
      '編解碼器權重精度："int8" 表示把編解碼器 transformer 的 GEMM 權重存成'
      + ' int8 分組 32 的仿射量化（常駐占用約減半，音質代價很小）；預設/留'
      + '空 = f16'],
  'cfg.text-to-speech.max_new_tokens': ['',
      '仅 8B：每个节拍的延迟模式生成预算（>= 1）',
      '僅 8B：每個節拍的延遲模式生成預算（>= 1）'],
  'cfg.text-to-speech.stream_chunk_frames': ['',
      '每生成 N 个编解码帧就发出一段 PCM（准实时流式）；0 = 一次性（单个节'
      + '拍）。每帧约 80 毫秒音频。',
      '每產生 N 個編解碼影格就發出一段 PCM（準即時串流）；0 = 一次性（單個'
      + '節拍）。每格約 80 毫秒音訊。'],
  'cfg.text-to-speech.interrupt_on_new_text': ['',
      '新文本一到就中止正在合成的这句话（先把已产生的部分冲刷出去），然后朗'
      + '读新文本；false = 每句话都完整念完再念下一句',
      '新文字一到就中止正在合成的這句話（先把已產生的部分沖出），然後朗讀新'
      + '文字；false = 每句話都完整念完再念下一句'],
  'cfg.text-to-speech.max_frames': ['',
      '仅 v1.5：每个节拍的帧预算（每帧约 80 毫秒 @ 48 kHz）；>= 1',
      '僅 v1.5：每個節拍的影格預算（每格約 80 毫秒 @ 48 kHz）；>= 1'],
  'cfg.text-to-speech.instruction': ['',
      '仅 v1.5：可选的风格指令（prompt 字段）',
      '僅 v1.5：可選的風格指令（prompt 欄位）'],
  'cfg.text-to-speech.language': ['',
      '仅 v1.5：可选的语言标签（prompt 字段）',
      '僅 v1.5：可選的語言標籤（prompt 欄位）'],
  'cfg.text-to-speech.voice_lock': ['',
      '一次定音：缓存第一次生成出来的音色，并在之后每个节拍都把它当作克隆参'
      + '考，使不同文本之间音色保持一致（第一次的音色由音频采样器的种子决定'
      + '）。audio 输入端口上的参考会覆盖它。不需要任何音频输入。',
      '一次定音：快取第一次產生出來的音色，並在之後每個節拍都把它當作複製參'
      + '考，使不同文字之間音色保持一致（第一次的音色由音訊取樣器的種子決定'
      + '）。audio 輸入埠上的參考會覆寫它。不需要任何音訊輸入。'],
  'cfg.text-to-speech.voice_ref_seconds': ['',
      '用于克隆的参考音频最多保留多少秒（越长每个节拍的提示开销越大）；对输'
      + '入端口的参考和 voice_lock 都适用。小于等于 0 表示保留整段。',
      '用於複製的參考音訊最多保留多少秒（越長每個節拍的提示開銷越大）；對輸'
      + '入埠的參考和 voice_lock 都適用。小於等於 0 表示保留整段。'],
  'port.text-to-speech.text': ['',
      'FlexData 字符串（或带 "text" 键的对象）：要合成的文本。对象还可以携'
      + '带 "end_of_response"（布尔）；只有当前这句话的 end_of_response 为 '
      + 'true（或缺失，默认视为 true）时，新节拍才会打断它。来自 text-chat '
      + '流式输出端口的中途片段（false）总是被完整朗读',
      'FlexData 字串（或帶 "text" 鍵的物件）：要合成的文字。物件還可以攜帶 '
      + '"end_of_response"（布林）；只有當前這句話的 end_of_response 為 '
      + 'true（或缺少，預設視為 true）時，新節拍才會打斷它。來自 text-chat '
      + '串流輸出埠的中途片段（false）總是被完整朗讀'],
  'port.text-to-speech.audio-ref': ['',
      '可选的单声道 f32 PCM TensorBeat（采样率不限；sideband.sample_rate 会'
      + '被采用）：要克隆的参考声音。最新的节拍具有粘性（在被替换之前对之后'
      + '所有句子都生效）。它自成一个时钟组——它的到来独立于文本流与 PCM 输'
      + '出，并不与它们锁定速率。',
      '可選的單聲道 f32 PCM TensorBeat（取樣率不限；sideband.sample_rate 會'
      + '被採用）：要複製的參考聲音。最新的節拍具有黏性（在被取代之前對之後'
      + '所有句子都生效）。它自成一個時脈組——它的到來獨立於文字串流與 PCM '
      + '輸出，並不與它們鎖定速率。'],
  'port.text-to-speech.audio-sampler': ['',
      '可选的 token 采样器配置 FlexData（sampler-select），用于音频码通道；'
      + '它的 seed 同时也是音色种子。在第一个节拍时锁存。未接线则沿用 MOSS '
      + '自己推荐的采样设置（不是贪心——贪心的音频会退化成无声的循环）',
      '可選的 token 取樣器組態 FlexData（sampler-select），用於音訊碼通道；'
      + '它的 seed 同時也是音色種子。在第一個節拍時鎖存。未接線則沿用 MOSS '
      + '自己建議的取樣設定（不是貪婪——貪婪的音訊會退化成無聲的迴圈）'],
  'port.text-to-speech.text-sampler': ['',
      '可选的 token 采样器配置 FlexData（sampler-select），用于自由文本通道'
      + '。在第一个节拍时锁存；未接线 = 贪心，而这里贪心正是合适的默认（文'
      + '本通道由 vpipe 生成，它必须紧跟转写内容才能到达 audio_end）',
      '可選的 token 取樣器組態 FlexData（sampler-select），用於自由文字通道'
      + '。在第一個節拍時鎖存；未接線 = 貪婪，而這裡貪婪正是合適的預設（文'
      + '字通道由 vpipe 產生，它必須緊跟轉寫內容才能抵達 audio_end）'],
  'port.text-to-speech.pcm': ['',
      'f32 PCM 的 TensorBeat（已设置 sideband.sample_rate）；下游可选。8B '
      + '变体：秩为 1 的 [采样数] 单声道 @ 24 kHz。v1.5 变体：秩为 2 的 [2,'
      + ' 采样数] 立体声 @ 48 kHz。',
      'f32 PCM 的 TensorBeat（已設定 sideband.sample_rate）；下游可選。8B '
      + '變體：秩為 1 的 [取樣數] 單聲道 @ 24 kHz。v1.5 變體：秩為 2 的 [2,'
      + ' 取樣數] 立體聲 @ 48 kHz。'],

  // ---- VAE Decode (generative) ----
  'stage.vae-decode.name': ['', 'VAE 解码', 'VAE 解碼'],
  'stage.vae-decode.doc': ['',
      '在 metal-compute 后端把 VAE 潜变量解码为平面 U8 RGB——生成/解码拆分中'
      + '的后半段，图像和视频都适用。图像潜变量产生一个节拍；视频潜变量则每'
      + '帧一个节拍，好让下游沿用已有的逐帧机制（save-image、rgb-to-video -'
      + '> save-video、预览），也可以在 1 号输出端口以单个节拍取回整段片段'
      + '。具体运行哪个解码器取决于常驻模型族；由插件注册的模型族自行解码。',
      '在 metal-compute 後端把 VAE 潛變數解碼為平面 U8 RGB——生成/解碼拆分中'
      + '的後半段，影像和影片都適用。影像潛變數產生一個節拍；影片潛變數則每'
      + '格一個節拍，好讓下游沿用既有的逐格機制（save-image、rgb-to-video -'
      + '> save-video、預覽），也可以在 1 號輸出埠以單個節拍取回整段片段。'
      + '具體執行哪個解碼器取決於常駐模型族；由外掛註冊的模型族自行解碼。'],
  'cfg.vae-decode.hf_dir': ['',
      '要解码其 VAE 的模型根目录；常驻模型族会在其中定位 VAE（惯例是 <'
      + 'hf_dir>/vae）。可选：model 输入端口上的 model-select 源会覆盖它',
      '要解碼其 VAE 的模型根目錄；常駐模型族會在其中定位 VAE（慣例是 <'
      + 'hf_dir>/vae）。可選：model 輸入埠上的 model-select 來源會覆寫它'],
  'cfg.vae-decode.fps': ['',
      '当潜变量本身没有携带帧率时，打在每个解码出的视频帧边带上的帧率。来自'
      + ' generate-video 的潜变量是携带的，因此这只是从别处读入时的后备值。'
      + '潜变量是图像时忽略',
      '當潛變數本身沒有攜帶影格率時，打在每個解碼出的影片影格邊帶上的影格率'
      + '。來自 generate-video 的潛變數是攜帶的，因此這只是從別處讀入時的後'
      + '備值。潛變數是影像時忽略'],
  'cfg.vae-decode.unload_when_idle': ['',
      '每个节拍处理完就释放 VAE 权重，下一个节拍再重新加载。本阶段在整个去'
      + '噪过程中都空闲，因此在内存吃紧的机器上，把它（权重以及解码工作集）'
      + '释放掉，正是让一个大 DiT 能在同一台机器上跑起来的关键。"auto"（默'
      + '认）根据物理内存与流水线的权重字节数自行判断；"always" / "never" '
      + '可强制',
      '每個節拍處理完就釋放 VAE 權重，下一個節拍再重新載入。本階段在整個去'
      + '噪過程中都閒置，因此在記憶體吃緊的機器上，把它（權重以及解碼工作集'
      + '）釋放掉，正是讓一個大 DiT 能在同一台機器上跑起來的關鍵。"auto"（'
      + '預設）根據實體記憶體與管線的權重位元組數自行判斷；"always" / "'
      + 'never" 可強制'],
  'port.vae-decode.latent': ['',
      'f32 潜变量，已解包并白化：图像为 [z, H/r, W/r]，片段为 [z, T, H/r, W'
      + '/r]。z 与空间步幅 r 即常驻模型族的 VAE 几何参数',
      'f32 潛變數，已解包並白化：影像為 [z, H/r, W/r]，片段為 [z, T, H/r, W'
      + '/r]。z 與空間步幅 r 即常駐模型族的 VAE 幾何參數'],
  'port.vae-decode.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.vae-decode.image': ['',
      '解码出的平面 U8 RGB TensorBeat [3, H, W]。视频潜变量按帧逐个发出节拍'
      + '，而不是发一个片段形状的张量，每个节拍的边带都带有 {frame, frames,'
      + ' fps}，让消费者知道它位于片段中的什么位置、后面还有多少',
      '解碼出的平面 U8 RGB TensorBeat [3, H, W]。影片潛變數按格逐個發出節拍'
      + '，而不是發一個片段形狀的張量，每個節拍的邊帶都帶有 {frame, frames,'
      + ' fps}，讓消費者知道它位於片段中的什麼位置、後面還有多少'],
  'port.vae-decode.clip': ['',
      '可选：同样的像素以单个节拍给出，平面 U8 RGB [frames, 3, H, W]，边带'
      + '带 {frames, fps}——正是 temporal-stack 构建的那种形状，也是参考条件'
      + '模型所需要的。只对视频潜变量、且只在接了消费者时才写出，因为构建它'
      + '要复制整段片段。0 号输出端口不受影响，因此需要逐帧的图无需再拆包',
      '可選：同樣的像素以單個節拍給出，平面 U8 RGB [frames, 3, H, W]，邊帶'
      + '帶 {frames, fps}——正是 temporal-stack 建構的那種形狀，也是參考條件'
      + '模型所需要的。只對影片潛變數、且只在接了消費者時才寫出，因為建構它'
      + '要複製整段片段。0 號輸出埠不受影響，因此需要逐格的圖無需再拆包'],

  // ---- VAE Encode (generative) ----
  'stage.vae-encode.name': ['', 'VAE 编码', 'VAE 編碼'],
  'stage.vae-encode.doc': ['',
      '在 metal-compute 后端把 RGB 图像编码为 Krea-2-Turbo（Qwen-Image VAE'
      + '）的白化潜变量。它是 vae-decode 的镜像；下游接 generate-image 的 `'
      + 'latent` 端口（图生图）。',
      '在 metal-compute 後端把 RGB 影像編碼為 Krea-2-Turbo（Qwen-Image VAE'
      + '）的白化潛變數。它是 vae-decode 的鏡像；下游接 generate-image 的 `'
      + 'latent` 埠（圖生圖）。'],
  'cfg.vae-encode.input_range': ['',
      '仅对 F32 输入有效（U8 恒为 0..255）：采样值代表什么。"unit"（默认）'
      + '是 [0,1]——即 video-to-rgb 打开 `normalize` 时发出的范围，也是预览'
      + '阶段所说的归一化；"byte" 是 [0,255]，即 video-to-rgb 关闭 `'
      + 'normalize` 时；"signed" 是 [-1,1]，VAE 自身的约定，供已经完成映射'
      + '的生产者使用。设错不会报错——它只会让每个潜变量整体偏移，取回的画面'
      + '看起来不过是发白而已',
      '僅對 F32 輸入有效（U8 恆為 0..255）：取樣值代表什麼。"unit"（預設）'
      + '是 [0,1]——即 video-to-rgb 開啟 `normalize` 時發出的範圍，也是預覽'
      + '階段所說的正規化；"byte" 是 [0,255]，即 video-to-rgb 關閉 `'
      + 'normalize` 時；"signed" 是 [-1,1]，VAE 自身的約定，供已經完成映射'
      + '的生產者使用。設錯不會報錯——它只會讓每個潛變數整體偏移，取回的畫面'
      + '看起來不過是發白而已'],
  'cfg.vae-encode.frames': ['',
      '仅 WAN，且仅对单张图片的节拍有效：图生视频的条件片段跨越多少帧。该潜'
      + '变量编码的是“条件图像后接这个数目减一的空白帧”，而不是单张图像——因'
      + '为 VAE 的时间卷积会混合相邻帧，所以 1 帧的编码结果是另一个张量。它'
      + '必须与 generate-video 阶段的 `frames` 一致；两者向上取整的方式相同'
      + '，所以填同一个数就够了。若送入的是堆叠片段，则前面的帧用真实帧而非'
      + '空白帧填充，并截断到这个长度，因为它所条件化的潜变量形状取自 DiT '
      + '阶段的 `frames`，而不是取自参考素材',
      '僅 WAN，且僅對單張圖片的節拍有效：圖生影片的條件片段跨越多少影格。該'
      + '潛變數編碼的是「條件影像後接這個數目減一的空白影格」，而不是單張影'
      + '像——因為 VAE 的時間卷積會混合相鄰影格，所以 1 格的編碼結果是另一個'
      + '張量。它必須與 generate-video 階段的 `frames` 一致；兩者向上取整的'
      + '方式相同，所以填同一個數就夠了。若送入的是堆疊片段，則前面的影格用'
      + '真實影格而非空白影格填充，並截斷到這個長度，因為它所條件化的潛變數'
      + '形狀取自 DiT 階段的 `frames`，而不是取自參考素材'],
  'cfg.vae-encode.hf_dir': ['',
      'Krea-2-Turbo / FLUX.2 / Qwen-Image-Edit / Mage-Flow 模型目录（VAE 从'
      + ' <hf_dir>/vae 读取）。可选：model 输入端口上的 model-select 源会覆'
      + '盖它',
      'Krea-2-Turbo / FLUX.2 / Qwen-Image-Edit / Mage-Flow 模型目錄（VAE 從'
      + ' <hf_dir>/vae 讀取）。可選：model 輸入埠上的 model-select 來源會覆'
      + '寫它'],
  'cfg.vae-encode.target_width': ['',
      '编码前先把输入按信箱模式缩放到这个宽度（须为 8 的倍数；需同时设置 '
      + 'target_height）。0/未设置 = 按原始尺寸编码',
      '編碼前先把輸入按信箱模式縮放到這個寬度（須為 8 的倍數；需同時設定 '
      + 'target_height）。0/未設定 = 按原始尺寸編碼'],
  'cfg.vae-encode.target_height': ['',
      '信箱模式缩放的目标高度（须为 8 的倍数；需同时设置 target_width）',
      '信箱模式縮放的目標高度（須為 8 的倍數；需同時設定 target_width）'],
  'cfg.vae-encode.pad_color': ['',
      '信箱模式的填充色：[r,g,b] 0..255、颜色名（black/white/gray），或单个'
      + '灰度值。默认黑色',
      '信箱模式的填充色：[r,g,b] 0..255、顏色名（black/white/gray），或單個'
      + '灰階值。預設黑色'],
  'cfg.vae-encode.unload_when_idle': ['',
      '每个节拍处理完就释放 VAE 权重，下一个节拍再重新加载。本阶段在整个去'
      + '噪过程中都空闲，因此在内存吃紧的机器上，把它（权重以及解码工作集）'
      + '释放掉，正是让一个大 DiT 能在同一台机器上跑起来的关键。"auto"（默'
      + '认）根据物理内存与流水线的权重字节数自行判断；"always" / "never" '
      + '可强制',
      '每個節拍處理完就釋放 VAE 權重，下一個節拍再重新載入。本階段在整個去'
      + '噪過程中都閒置，因此在記憶體吃緊的機器上，把它（權重以及解碼工作集'
      + '）釋放掉，正是讓一個大 DiT 能在同一台機器上跑起來的關鍵。"auto"（'
      + '預設）根據實體記憶體與管線的權重位元組數自行判斷；"always" / "'
      + 'never" 可強制'],
  'port.vae-encode.image': ['',
      'U8 或 f32 RGB，通道优先（U8 为 0..255；f32 按 `input_range` 解读，默'
      + '认 [0,1]）。由秩决定含义：[3,H,W] 是一张图片，[frames,3,H,W] 是一'
      + '段片段——即 temporal-stack 发出的形状，也是视频 VAE 一次调用就能编'
      + '码的形状，因为它是因果的。图像 VAE 遇到片段会拒绝，而不是只取它的'
      + '第一帧',
      'U8 或 f32 RGB，通道優先（U8 為 0..255；f32 按 `input_range` 解讀，預'
      + '設 [0,1]）。由秩決定含義：[3,H,W] 是一張圖片，[frames,3,H,W] 是一'
      + '段片段——即 temporal-stack 發出的形狀，也是影片 VAE 一次呼叫就能編'
      + '碼的形狀，因為它是因果的。影像 VAE 遇到片段會拒絕，而不是只取它的'
      + '第一格'],
  'port.vae-encode.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.vae-encode.latent': ['',
      'f32 白化潜变量 [z_dim, H/8, W/8]（未打包）',
      'f32 白化潛變數 [z_dim, H/8, W/8]（未打包）'],

  // ---- Video Reference Encoder (generative) ----
  'stage.video-ref-encoder.name': ['', '视频参考编码器', '影片參考編碼器'],
  'stage.video-ref-encoder.doc': ['',
      'MiniMax-H3 ref2va 参考编码器：一组参考图像、片段与音轨（外加一条提示'
      + '词）-> 视频 DiT 据以去噪的条件与参考潜变量行。之所以用列表而不是每'
      + '个参考一个端口作为输入，是因为一次请求会带上数量不定的十来个参考；'
      + '每一个都在这里解码，好让它的帧率和采样率传达给负责重采样到 24 fps '
      + '的模型。请与同一个 hf_dir、同样 `frames` 的 generate-video 阶段配'
      + '对使用。',
      'MiniMax-H3 ref2va 參考編碼器：一組參考影像、片段與音軌（外加一條提示'
      + '詞）-> 影片 DiT 據以去噪的條件與參考潛變數列。之所以用清單而不是每'
      + '個參考一個埠作為輸入，是因為一次請求會帶上數量不定的十來個參考；每'
      + '一個都在這裡解碼，好讓它的影格率和取樣率傳達給負責重新取樣到 24 '
      + 'fps 的模型。請與同一個 hf_dir、同樣 `frames` 的 generate-video 階'
      + '段配對使用。'],
  'cfg.video-ref-encoder.references': ['',
      '参考文件，按模型应当读取的顺序给出：一个路径，或一个路径数组。顺序决'
      + '定它们在呈现中的标号（<Picture 1>、<Video 2>…），也决定它们在共享'
      + '旋转时钟上的排布，因此顺序不同就是另一个请求。每个文件是什么种类——'
      + '图像、视频还是音频——从文件内容而不是文件名读出：只有一帧的容器算图'
      + '像参考，没有视频流的 .mp4 算音频参考。最多 9 个图像、3 个视频、3 '
      + '个音频，总计 12 个，且音频不能是唯一的种类',
      '參考檔案，按模型應當讀取的順序給出：一個路徑，或一個路徑陣列。順序決'
      + '定它們在呈現中的標號（<Picture 1>、<Video 2>…），也決定它們在共享'
      + '旋轉時鐘上的排布，因此順序不同就是另一個請求。每個檔案是什麼種類——'
      + '影像、影片還是音訊——從檔案內容而不是檔名讀出：只有一格的容器算影像'
      + '參考，沒有視訊串流的 .mp4 算音訊參考。最多 9 個影像、3 個影片、3 '
      + '個音訊，總計 12 個，且音訊不能是唯一的種類'],
  'cfg.video-ref-encoder.hf_dir': ['',
      'MiniMax-H3 模型目录（text_encoder/、video_vae/、audio_vae/）。可选：'
      + 'model 输入端口上的 model-select 源会覆盖它',
      'MiniMax-H3 模型目錄（text_encoder/、video_vae/、audio_vae/）。可選：'
      + 'model 輸入埠上的 model-select 來源會覆寫它'],
  'cfg.video-ref-encoder.frames': ['',
      '以 24 fps 计的生成帧数，向上取整到下一个 17n+5。必须与 '
      + 'generate-video 阶段的 `frames` 一致：所有参考都会被截断到这个时长'
      + '，DiT 构建的布局也是按同一个数字定尺寸的',
      '以 24 fps 計的生成影格數，向上取整到下一個 17n+5。必須與 '
      + 'generate-video 階段的 `frames` 一致：所有參考都會被截斷到這個長度'
      + '，DiT 建構的版面也是按同一個數字定尺寸的'],
  'cfg.video-ref-encoder.reference_image_short_edge': ['',
      '图像参考的短边被缩放到多少（发布检查点为 2048）。与目标画布不同，这'
      + '一项没有面积上限，且会放大小图——图像参考是按高细节读取的，永远不会'
      + '限定所生成的画面几何',
      '影像參考的短邊被縮放到多少（發布檢查點為 2048）。與目標畫布不同，這'
      + '一項沒有面積上限，且會放大小圖——影像參考是按高細節讀取的，永遠不會'
      + '限定所產生的畫面幾何'],
  'cfg.video-ref-encoder.reference_image_max_pixels': ['',
      '图像参考的面积上限，单位像素。0（默认）表示不限，这正是发布检查点的'
      + '规则——图像按高细节读取，`reference_image_short_edge` 是它唯一的界'
      + '限。可一旦短边设为 0，表示按送来的尺寸取图，那道界限就不复存在，因'
      + '此把原始图片直接送进参考输入端口的图应当设置这一项：一张不限尺寸的'
      + ' 4K 静态图约合 220 个 VAE 分块和 8160 个 DiT 行，相当于一次典型请'
      + '求打包序列的一半，另外还要加上相应的视觉 token',
      '影像參考的面積上限，單位像素。0（預設）表示不限，這正是發布檢查點的'
      + '規則——影像按高細節讀取，`reference_image_short_edge` 是它唯一的界'
      + '限。可一旦短邊設為 0，表示按送來的尺寸取圖，那道界限就不復存在，因'
      + '此把原始圖片直接送進參考輸入埠的圖應當設定這一項：一張不限尺寸的 '
      + '4K 靜態圖約合 220 個 VAE 分塊和 8160 個 DiT 列，相當於一次典型請求'
      + '打包序列的一半，另外還要加上相應的視覺 token'],
  'cfg.video-ref-encoder.reference_video_short_edge': ['',
      '视频参考在编码前其短边被缩放到多少（发布检查点为 768），并受 `'
      + 'reference_video_max_pixels` 限制。0 表示用片段自身的短边——同一条规'
      + '则，只是从文件本身的尺寸出发，因此片段永远不会被放大。设为 768 时'
      + '，一段 960x544 的参考会落到 1344x768 的画布上，像素数是原来的 1.98'
      + ' 倍，而信息量并没有增加。调低它会同时影响三处：VAE 编码、DiT 每一'
      + '步都要注意的参考行，以及最容易被忽略的条件生成——因为视觉塔拿到的正'
      + '是这些像素，于是该片段贡献的 token 也随之减少。在双参考示例上实测'
      + '，768 -> 0：参考行 13120 -> 7144，条件行 3115 -> 2119，耗时 23 分 '
      + '07 秒 -> 14 分 17 秒。这是一个同时影响两条通路的保真度取舍，而不是'
      + '白捡的提速，因此定下来之前请先对比效果',
      '影片參考在編碼前其短邊被縮放到多少（發布檢查點為 768），並受 `'
      + 'reference_video_max_pixels` 限制。0 表示用片段自身的短邊——同一條規'
      + '則，只是從檔案本身的尺寸出發，因此片段永遠不會被放大。設為 768 時'
      + '，一段 960x544 的參考會落到 1344x768 的畫布上，像素數是原來的 1.98'
      + ' 倍，而資訊量並沒有增加。調低它會同時影響三處：VAE 編碼、DiT 每一'
      + '步都要注意的參考列，以及最容易被忽略的條件產生——因為視覺塔拿到的正'
      + '是這些像素，於是該片段貢獻的 token 也隨之減少。在雙參考範例上實測'
      + '，768 -> 0：參考列 13120 -> 7144，條件列 3115 -> 2119，耗時 23 分 '
      + '07 秒 -> 14 分 17 秒。這是一個同時影響兩條通路的保真度取捨，而不是'
      + '白撿的提速，因此定下來之前請先對比效果'],
  'cfg.video-ref-encoder.reference_video_max_pixels': ['',
      '视频参考的画布所受的面积上限，单位像素（发布检查点为 768 * 1344）。'
      + '它在短边缩放之后、取整之前生效，因此无论 `'
      + 'reference_video_short_edge` 取何值，比画布更大的参考都由它来约束',
      '影片參考的畫布所受的面積上限，單位像素（發布檢查點為 768 * 1344）。'
      + '它在短邊縮放之後、取整之前生效，因此無論 `'
      + 'reference_video_short_edge` 取何值，比畫布更大的參考都由它來約束'],
  'cfg.video-ref-encoder.attach_audio': ['',
      '参考输入端口编号（1..6），表示这些端口上的音频是前一个参考的配乐，而'
      + '不是独立的参考——这样一段被解复用成画面与 PCM 的片段虽然从两个端口'
      + '送入，却仍算一个参考，一并标为 <Video k> 与 <Audio j>，与带配乐的'
      + '文件完全一致。它是位置相关的：附加到紧邻其前的那个参考上，而那个参'
      + '考也可能来自 `references` 列表，因为列表是先读的。之所以逐端口设置'
      + '而不是一个总开关，是因为一次请求完全可能既带随片配乐、又带一段独立'
      + '音乐。节拍自身的 `attach` 边带可针对该节拍覆盖此项',
      '參考輸入埠編號（1..6），表示這些埠上的音訊是前一個參考的配樂，而不是'
      + '獨立的參考——這樣一段被解多工成畫面與 PCM 的片段雖然從兩個埠送入，'
      + '卻仍算一個參考，一併標為 <Video k> 與 <Audio j>，與帶配樂的檔案完'
      + '全一致。它是位置相關的：附加到緊鄰其前的那個參考上，而那個參考也可'
      + '能來自 `references` 清單，因為清單是先讀的。之所以逐埠設定而不是一'
      + '個總開關，是因為一次請求完全可能既帶隨片配樂、又帶一段獨立音樂。節'
      + '拍自身的 `attach` 邊帶可針對該節拍覆寫此項'],
  'cfg.video-ref-encoder.video_sample_fps': ['',
      '条件生成读取视频参考所用的速率——每 24/该值 帧取一帧，两两合并成带时'
      + '间戳的视觉块。它不是 VAE 编码所用的速率，后者是完整的 24',
      '條件產生讀取影片參考所用的速率——每 24/該值 格取一格，兩兩合併成帶時'
      + '間戳記的視覺區塊。它不是 VAE 編碼所用的速率，後者是完整的 24'],
  'cfg.video-ref-encoder.max_prompt_tokens': ['',
      '条件生成的序列池大小。ref2va 的呈现比纯文本提示词长得多——单个短边 '
      + '2048 的图像参考就会贡献数千个视觉 token——因此文本路径默认的 4096 '
      + '连一个都装不下。它要占 KV：被抽取的 50 层每个 token 约 200 KB，所'
      + '以 16384 意味着条件生成常驻期间约占 3.3 GB。参考图像很多的请求可以'
      + '调高；机器内存小则调低',
      '條件產生的序列池大小。ref2va 的呈現比純文字提示詞長得多——單個短邊 '
      + '2048 的影像參考就會貢獻數千個視覺 token——因此文字路徑預設的 4096 '
      + '連一個都裝不下。它要占 KV：被抽取的 50 層每個 token 約 200 KB，所'
      + '以 16384 意味著條件產生常駐期間約占 3.3 GB。參考影像很多的請求可以'
      + '調高；機器記憶體小則調低'],
  'cfg.video-ref-encoder.unload_when_idle': ['',
      '参考编码完成后释放条件生成器与各 VAE，下一次请求再重新加载。32B 的条'
      + '件生成器是 ref2va 图中最大的常驻块，而它在整个去噪过程中都空闲，在'
      + '本模型上那是好几分钟。"auto"（默认）根据物理内存与流水线的权重字节'
      + '数自行判断；"always" / "never" 可强制',
      '參考編碼完成後釋放條件產生器與各 VAE，下一次請求再重新載入。32B 的條'
      + '件產生器是 ref2va 圖中最大的常駐區塊，而它在整個去噪過程中都閒置，'
      + '在本模型上那是好幾分鐘。"auto"（預設）根據實體記憶體與管線的權重位'
      + '元組數自行判斷；"always" / "never" 可強制'],
  'port.video-ref-encoder.prompt': ['',
      '提示词文本（FlexData 字符串或 {text: ...}），按原文使用——不套聊天模'
      + '板，也不加特殊 token',
      '提示詞文字（FlexData 字串或 {text: ...}），按原文使用——不套聊天範本'
      + '，也不加特殊 token'],
  'port.video-ref-encoder.model': ['',
      '可选的共享模型引用，来自 model-select 源；覆盖 hf_dir 配置',
      '可選的共享模型參照，來自 model-select 來源；覆寫 hf_dir 設定'],
  'port.video-ref-encoder.ref1': ['',
      '可选的张量形式参考素材，每个参考一个节拍，编号排在 `references` 列表'
      + '之后。种类由张量的秩决定：[N] 或 [声道数, N] 的 f32 是音频，[3, H,'
      + ' W] 的 u8 是图片，[frames, 3, H, W] 的 u8 是片段——这也解决了文件本'
      + '身无法表达的情形，因为单帧片段是 [1, 3, H, W]，而静态图是 [3, H, W'
      + ']。边带携带各种速率，缺少它们是错误而不是走默认值：片段需要 `fps`'
      + '，音频需要 `sr`（或 `sample_rate`）——采样率与音频 VAE 不一致的音轨'
      + '会被重采样到 VAE 的采样率，这正是要求标明采样率的原因，所以请把生'
      + '产者的采样率设成 VAE 的（发布检查点为 32000），以免重采样两次。可'
      + '选的边带键：`short_edge` 指定这一个参考的画布（此处默认 0，表示按'
      + '送来的尺寸编码）；音频节拍上的 `attach` 可针对该节拍覆盖本阶段的 `'
      + 'attach_audio`——true 表示把它作为前一个参考的配乐，false 表示拒绝配'
      + '置中提出的附加。已接线的端口每次请求都必须发节拍——空张量就是“本次'
      + '没有内容”的说法，因为一个沉默的端口会让其后所有参考重新编号，从而'
      + '悄悄改变整个请求',
      '可選的張量形式參考素材，每個參考一個節拍，編號排在 `references` 清單'
      + '之後。種類由張量的秩決定：[N] 或 [聲道數, N] 的 f32 是音訊，[3, H,'
      + ' W] 的 u8 是圖片，[frames, 3, H, W] 的 u8 是片段——這也解決了檔案本'
      + '身無法表達的情形，因為單格片段是 [1, 3, H, W]，而靜態圖是 [3, H, W'
      + ']。邊帶攜帶各種速率，缺少它們是錯誤而不是走預設值：片段需要 `fps`'
      + '，音訊需要 `sr`（或 `sample_rate`）——取樣率與音訊 VAE 不一致的音軌'
      + '會被重新取樣到 VAE 的取樣率，這正是要求標明取樣率的原因，所以請把'
      + '生產者的取樣率設成 VAE 的（發布檢查點為 32000），以免重新取樣兩次'
      + '。可選的邊帶鍵：`short_edge` 指定這一個參考的畫布（此處預設 0，表'
      + '示按送來的尺寸編碼）；音訊節拍上的 `attach` 可針對該節拍覆寫本階段'
      + '的 `attach_audio`——true 表示把它作為前一個參考的配樂，false 表示拒'
      + '絕設定中提出的附加。已接線的埠每次請求都必須發節拍——空張量就是「本'
      + '次沒有內容」的說法，因為一個沉默的埠會讓其後所有參考重新編號，從而'
      + '悄悄改變整個請求'],
  'port.video-ref-encoder.ref2': ['',
      '可选的张量形式参考素材，每个参考一个节拍，编号排在 `references` 列表'
      + '之后。种类由张量的秩决定：[N] 或 [声道数, N] 的 f32 是音频，[3, H,'
      + ' W] 的 u8 是图片，[frames, 3, H, W] 的 u8 是片段——这也解决了文件本'
      + '身无法表达的情形，因为单帧片段是 [1, 3, H, W]，而静态图是 [3, H, W'
      + ']。边带携带各种速率，缺少它们是错误而不是走默认值：片段需要 `fps`'
      + '，音频需要 `sr`（或 `sample_rate`）——采样率与音频 VAE 不一致的音轨'
      + '会被重采样到 VAE 的采样率，这正是要求标明采样率的原因，所以请把生'
      + '产者的采样率设成 VAE 的（发布检查点为 32000），以免重采样两次。可'
      + '选的边带键：`short_edge` 指定这一个参考的画布（此处默认 0，表示按'
      + '送来的尺寸编码）；音频节拍上的 `attach` 可针对该节拍覆盖本阶段的 `'
      + 'attach_audio`——true 表示把它作为前一个参考的配乐，false 表示拒绝配'
      + '置中提出的附加。已接线的端口每次请求都必须发节拍——空张量就是“本次'
      + '没有内容”的说法，因为一个沉默的端口会让其后所有参考重新编号，从而'
      + '悄悄改变整个请求',
      '可選的張量形式參考素材，每個參考一個節拍，編號排在 `references` 清單'
      + '之後。種類由張量的秩決定：[N] 或 [聲道數, N] 的 f32 是音訊，[3, H,'
      + ' W] 的 u8 是圖片，[frames, 3, H, W] 的 u8 是片段——這也解決了檔案本'
      + '身無法表達的情形，因為單格片段是 [1, 3, H, W]，而靜態圖是 [3, H, W'
      + ']。邊帶攜帶各種速率，缺少它們是錯誤而不是走預設值：片段需要 `fps`'
      + '，音訊需要 `sr`（或 `sample_rate`）——取樣率與音訊 VAE 不一致的音軌'
      + '會被重新取樣到 VAE 的取樣率，這正是要求標明取樣率的原因，所以請把'
      + '生產者的取樣率設成 VAE 的（發布檢查點為 32000），以免重新取樣兩次'
      + '。可選的邊帶鍵：`short_edge` 指定這一個參考的畫布（此處預設 0，表'
      + '示按送來的尺寸編碼）；音訊節拍上的 `attach` 可針對該節拍覆寫本階段'
      + '的 `attach_audio`——true 表示把它作為前一個參考的配樂，false 表示拒'
      + '絕設定中提出的附加。已接線的埠每次請求都必須發節拍——空張量就是「本'
      + '次沒有內容」的說法，因為一個沉默的埠會讓其後所有參考重新編號，從而'
      + '悄悄改變整個請求'],
  'port.video-ref-encoder.ref3': ['',
      '可选的张量形式参考素材，每个参考一个节拍，编号排在 `references` 列表'
      + '之后。种类由张量的秩决定：[N] 或 [声道数, N] 的 f32 是音频，[3, H,'
      + ' W] 的 u8 是图片，[frames, 3, H, W] 的 u8 是片段——这也解决了文件本'
      + '身无法表达的情形，因为单帧片段是 [1, 3, H, W]，而静态图是 [3, H, W'
      + ']。边带携带各种速率，缺少它们是错误而不是走默认值：片段需要 `fps`'
      + '，音频需要 `sr`（或 `sample_rate`）——采样率与音频 VAE 不一致的音轨'
      + '会被重采样到 VAE 的采样率，这正是要求标明采样率的原因，所以请把生'
      + '产者的采样率设成 VAE 的（发布检查点为 32000），以免重采样两次。可'
      + '选的边带键：`short_edge` 指定这一个参考的画布（此处默认 0，表示按'
      + '送来的尺寸编码）；音频节拍上的 `attach` 可针对该节拍覆盖本阶段的 `'
      + 'attach_audio`——true 表示把它作为前一个参考的配乐，false 表示拒绝配'
      + '置中提出的附加。已接线的端口每次请求都必须发节拍——空张量就是“本次'
      + '没有内容”的说法，因为一个沉默的端口会让其后所有参考重新编号，从而'
      + '悄悄改变整个请求',
      '可選的張量形式參考素材，每個參考一個節拍，編號排在 `references` 清單'
      + '之後。種類由張量的秩決定：[N] 或 [聲道數, N] 的 f32 是音訊，[3, H,'
      + ' W] 的 u8 是圖片，[frames, 3, H, W] 的 u8 是片段——這也解決了檔案本'
      + '身無法表達的情形，因為單格片段是 [1, 3, H, W]，而靜態圖是 [3, H, W'
      + ']。邊帶攜帶各種速率，缺少它們是錯誤而不是走預設值：片段需要 `fps`'
      + '，音訊需要 `sr`（或 `sample_rate`）——取樣率與音訊 VAE 不一致的音軌'
      + '會被重新取樣到 VAE 的取樣率，這正是要求標明取樣率的原因，所以請把'
      + '生產者的取樣率設成 VAE 的（發布檢查點為 32000），以免重新取樣兩次'
      + '。可選的邊帶鍵：`short_edge` 指定這一個參考的畫布（此處預設 0，表'
      + '示按送來的尺寸編碼）；音訊節拍上的 `attach` 可針對該節拍覆寫本階段'
      + '的 `attach_audio`——true 表示把它作為前一個參考的配樂，false 表示拒'
      + '絕設定中提出的附加。已接線的埠每次請求都必須發節拍——空張量就是「本'
      + '次沒有內容」的說法，因為一個沉默的埠會讓其後所有參考重新編號，從而'
      + '悄悄改變整個請求'],
  'port.video-ref-encoder.ref4': ['',
      '可选的张量形式参考素材，每个参考一个节拍，编号排在 `references` 列表'
      + '之后。种类由张量的秩决定：[N] 或 [声道数, N] 的 f32 是音频，[3, H,'
      + ' W] 的 u8 是图片，[frames, 3, H, W] 的 u8 是片段——这也解决了文件本'
      + '身无法表达的情形，因为单帧片段是 [1, 3, H, W]，而静态图是 [3, H, W'
      + ']。边带携带各种速率，缺少它们是错误而不是走默认值：片段需要 `fps`'
      + '，音频需要 `sr`（或 `sample_rate`）——采样率与音频 VAE 不一致的音轨'
      + '会被重采样到 VAE 的采样率，这正是要求标明采样率的原因，所以请把生'
      + '产者的采样率设成 VAE 的（发布检查点为 32000），以免重采样两次。可'
      + '选的边带键：`short_edge` 指定这一个参考的画布（此处默认 0，表示按'
      + '送来的尺寸编码）；音频节拍上的 `attach` 可针对该节拍覆盖本阶段的 `'
      + 'attach_audio`——true 表示把它作为前一个参考的配乐，false 表示拒绝配'
      + '置中提出的附加。已接线的端口每次请求都必须发节拍——空张量就是“本次'
      + '没有内容”的说法，因为一个沉默的端口会让其后所有参考重新编号，从而'
      + '悄悄改变整个请求',
      '可選的張量形式參考素材，每個參考一個節拍，編號排在 `references` 清單'
      + '之後。種類由張量的秩決定：[N] 或 [聲道數, N] 的 f32 是音訊，[3, H,'
      + ' W] 的 u8 是圖片，[frames, 3, H, W] 的 u8 是片段——這也解決了檔案本'
      + '身無法表達的情形，因為單格片段是 [1, 3, H, W]，而靜態圖是 [3, H, W'
      + ']。邊帶攜帶各種速率，缺少它們是錯誤而不是走預設值：片段需要 `fps`'
      + '，音訊需要 `sr`（或 `sample_rate`）——取樣率與音訊 VAE 不一致的音軌'
      + '會被重新取樣到 VAE 的取樣率，這正是要求標明取樣率的原因，所以請把'
      + '生產者的取樣率設成 VAE 的（發布檢查點為 32000），以免重新取樣兩次'
      + '。可選的邊帶鍵：`short_edge` 指定這一個參考的畫布（此處預設 0，表'
      + '示按送來的尺寸編碼）；音訊節拍上的 `attach` 可針對該節拍覆寫本階段'
      + '的 `attach_audio`——true 表示把它作為前一個參考的配樂，false 表示拒'
      + '絕設定中提出的附加。已接線的埠每次請求都必須發節拍——空張量就是「本'
      + '次沒有內容」的說法，因為一個沉默的埠會讓其後所有參考重新編號，從而'
      + '悄悄改變整個請求'],
  'port.video-ref-encoder.ref5': ['',
      '可选的张量形式参考素材，每个参考一个节拍，编号排在 `references` 列表'
      + '之后。种类由张量的秩决定：[N] 或 [声道数, N] 的 f32 是音频，[3, H,'
      + ' W] 的 u8 是图片，[frames, 3, H, W] 的 u8 是片段——这也解决了文件本'
      + '身无法表达的情形，因为单帧片段是 [1, 3, H, W]，而静态图是 [3, H, W'
      + ']。边带携带各种速率，缺少它们是错误而不是走默认值：片段需要 `fps`'
      + '，音频需要 `sr`（或 `sample_rate`）——采样率与音频 VAE 不一致的音轨'
      + '会被重采样到 VAE 的采样率，这正是要求标明采样率的原因，所以请把生'
      + '产者的采样率设成 VAE 的（发布检查点为 32000），以免重采样两次。可'
      + '选的边带键：`short_edge` 指定这一个参考的画布（此处默认 0，表示按'
      + '送来的尺寸编码）；音频节拍上的 `attach` 可针对该节拍覆盖本阶段的 `'
      + 'attach_audio`——true 表示把它作为前一个参考的配乐，false 表示拒绝配'
      + '置中提出的附加。已接线的端口每次请求都必须发节拍——空张量就是“本次'
      + '没有内容”的说法，因为一个沉默的端口会让其后所有参考重新编号，从而'
      + '悄悄改变整个请求',
      '可選的張量形式參考素材，每個參考一個節拍，編號排在 `references` 清單'
      + '之後。種類由張量的秩決定：[N] 或 [聲道數, N] 的 f32 是音訊，[3, H,'
      + ' W] 的 u8 是圖片，[frames, 3, H, W] 的 u8 是片段——這也解決了檔案本'
      + '身無法表達的情形，因為單格片段是 [1, 3, H, W]，而靜態圖是 [3, H, W'
      + ']。邊帶攜帶各種速率，缺少它們是錯誤而不是走預設值：片段需要 `fps`'
      + '，音訊需要 `sr`（或 `sample_rate`）——取樣率與音訊 VAE 不一致的音軌'
      + '會被重新取樣到 VAE 的取樣率，這正是要求標明取樣率的原因，所以請把'
      + '生產者的取樣率設成 VAE 的（發布檢查點為 32000），以免重新取樣兩次'
      + '。可選的邊帶鍵：`short_edge` 指定這一個參考的畫布（此處預設 0，表'
      + '示按送來的尺寸編碼）；音訊節拍上的 `attach` 可針對該節拍覆寫本階段'
      + '的 `attach_audio`——true 表示把它作為前一個參考的配樂，false 表示拒'
      + '絕設定中提出的附加。已接線的埠每次請求都必須發節拍——空張量就是「本'
      + '次沒有內容」的說法，因為一個沉默的埠會讓其後所有參考重新編號，從而'
      + '悄悄改變整個請求'],
  'port.video-ref-encoder.ref6': ['',
      '可选的张量形式参考素材，每个参考一个节拍，编号排在 `references` 列表'
      + '之后。种类由张量的秩决定：[N] 或 [声道数, N] 的 f32 是音频，[3, H,'
      + ' W] 的 u8 是图片，[frames, 3, H, W] 的 u8 是片段——这也解决了文件本'
      + '身无法表达的情形，因为单帧片段是 [1, 3, H, W]，而静态图是 [3, H, W'
      + ']。边带携带各种速率，缺少它们是错误而不是走默认值：片段需要 `fps`'
      + '，音频需要 `sr`（或 `sample_rate`）——采样率与音频 VAE 不一致的音轨'
      + '会被重采样到 VAE 的采样率，这正是要求标明采样率的原因，所以请把生'
      + '产者的采样率设成 VAE 的（发布检查点为 32000），以免重采样两次。可'
      + '选的边带键：`short_edge` 指定这一个参考的画布（此处默认 0，表示按'
      + '送来的尺寸编码）；音频节拍上的 `attach` 可针对该节拍覆盖本阶段的 `'
      + 'attach_audio`——true 表示把它作为前一个参考的配乐，false 表示拒绝配'
      + '置中提出的附加。已接线的端口每次请求都必须发节拍——空张量就是“本次'
      + '没有内容”的说法，因为一个沉默的端口会让其后所有参考重新编号，从而'
      + '悄悄改变整个请求',
      '可選的張量形式參考素材，每個參考一個節拍，編號排在 `references` 清單'
      + '之後。種類由張量的秩決定：[N] 或 [聲道數, N] 的 f32 是音訊，[3, H,'
      + ' W] 的 u8 是圖片，[frames, 3, H, W] 的 u8 是片段——這也解決了檔案本'
      + '身無法表達的情形，因為單格片段是 [1, 3, H, W]，而靜態圖是 [3, H, W'
      + ']。邊帶攜帶各種速率，缺少它們是錯誤而不是走預設值：片段需要 `fps`'
      + '，音訊需要 `sr`（或 `sample_rate`）——取樣率與音訊 VAE 不一致的音軌'
      + '會被重新取樣到 VAE 的取樣率，這正是要求標明取樣率的原因，所以請把'
      + '生產者的取樣率設成 VAE 的（發布檢查點為 32000），以免重新取樣兩次'
      + '。可選的邊帶鍵：`short_edge` 指定這一個參考的畫布（此處預設 0，表'
      + '示按送來的尺寸編碼）；音訊節拍上的 `attach` 可針對該節拍覆寫本階段'
      + '的 `attach_audio`——true 表示把它作為前一個參考的配樂，false 表示拒'
      + '絕設定中提出的附加。已接線的埠每次請求都必須發節拍——空張量就是「本'
      + '次沒有內容」的說法，因為一個沉默的埠會讓其後所有參考重新編號，從而'
      + '悄悄改變整個請求'],
  'port.video-ref-encoder.conditioning': ['',
      'bf16 [n_tokens, 5120]——Qwen3-VL-32B 第 50 层在整个呈现上的抽取结果。'
      + '与 diffusion-conditioner 发出的约定相同，因此 generate-video 两者'
      + '皆可接受；边带携带逐行的模态标签以及每个参考的潜变量几何',
      'bf16 [n_tokens, 5120]——Qwen3-VL-32B 第 50 層在整個呈現上的抽取結果。'
      + '與 diffusion-conditioner 發出的約定相同，因此 generate-video 兩者'
      + '皆可接受；邊帶攜帶逐列的模態標籤以及每個參考的潛變數幾何'],
  'port.video-ref-encoder.ref_video': ['',
      'f32 [rows, 96]——图像与视频参考的潜变量打包成的 DiT 行，按参考顺序拼'
      + '接。没有任何参考带视频时为 0 行',
      'f32 [rows, 96]——影像與影片參考的潛變數打包成的 DiT 列，按參考順序串'
      + '接。沒有任何參考帶影片時為 0 列'],
  'port.video-ref-encoder.ref_audio': ['',
      'f32 [rows, 32]——参考配乐，单个参考内部按声道优先排列，顺序与上面一致'
      + '。一个都没有时为 0 行',
      'f32 [rows, 32]——參考配樂，單個參考內部按聲道優先排列，順序與上面一致'
      + '。一個都沒有時為 0 列'],

  // ---- CoreML Model (generic) ----
  'stage.coreml-inference.name': ['', 'CoreML 模型', 'CoreML 模型'],
  'stage.coreml-inference.doc': ['',
      '在一个 TensorBeat 输入上运行通用 CoreML 模型，并为每个输出特征（'
      + 'output_feature_names）各发出一个 TensorBeat。',
      '在一個 TensorBeat 輸入上執行通用 CoreML 模型，並為每個輸出特徵（'
      + 'output_feature_names）各發出一個 TensorBeat。'],
  'cfg.coreml-inference.model_path': ['',
      '.mlmodelc 目录或 .mlmodel 文件',
      '.mlmodelc 目錄或 .mlmodel 檔案'],
  'cfg.coreml-inference.input_feature_name': ['',
      '模型输入特征名',
      '模型輸入特徵名'],
  'cfg.coreml-inference.output_feature_names': ['',
      '输出特征名的非空数组',
      '輸出特徵名的非空陣列'],
  'cfg.coreml-inference.compute_units': ['',
      '0=仅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE',
      '0=僅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE'],
  'cfg.coreml-inference.uses_cpu_only': ['',
      '推理时对 compute_units 的覆盖',
      '推理時對 compute_units 的覆寫'],
  'port.coreml-inference.input': ['',
      '作为 TensorBeat 的模型输入特征',
      '作為 TensorBeat 的模型輸入特徵'],

  // ---- passthrough (generic) ----
  'stage.passthrough.doc': ['',
      '恒等的 1 入 / 1 出阶段：原样转发每一个输入节拍。输入端的 EOS 结束本'
      + '阶段。',
      '恆等的 1 入 / 1 出階段：原樣轉發每一個輸入節拍。輸入端的 EOS 結束本'
      + '階段。'],
  'port.passthrough.in': ['', '任意节拍', '任意節拍'],
  'port.passthrough.out': ['', '原样转发的输入节拍', '原樣轉發的輸入節拍'],

  // ---- Boogu-Image Model Config (model-specific-config) ----
  'stage.boogu-image-model-config.name': ['',
      'Boogu-Image 模型配置',
      'Boogu-Image 模型設定'],
  'stage.boogu-image-model-config.doc': ['',
      '源：Boogu-Image 专属参数——它的管道对 VLM 条件图像施加的两个界限，两'
      + '者并不冗余。发一个节拍即结束；若接了 trigger 输入端口，则每收到一'
      + '个节拍发一次。',
      '來源：Boogu-Image 專屬參數——它的管道對 VLM 條件影像施加的兩個界限，'
      + '兩者並不冗餘。發一個節拍即結束；若接了 trigger 輸入埠，則每收到一'
      + '個節拍發一次。'],
  'cfg.boogu-image-model-config.vl_long_edge': ['',
      '接地编码：送入视觉塔之前，参考图像最长边的上限。未设置 => 768',
      '接地編碼：送入視覺塔之前，參考影像最長邊的上限。未設定 => 768'],
  'cfg.boogu-image-model-config.vl_pixel_budget': ['',
      '接地编码：总像素数上限，与 vl_long_edge 一同生效。未设置 => 本模型族'
      + '自己的 384x384（147456）',
      '接地編碼：總像素數上限，與 vl_long_edge 一同生效。未設定 => 本模型族'
      + '自己的 384x384（147456）'],
  'cfg.boogu-image-model-config.vl_min_pixels': ['',
      '接地编码：图像处理器的下界，正是它让过小或过于狭长的参考图在切 patch'
      + ' 前被放大。未设置 => 本模型族自己的 65536',
      '接地編碼：影像處理器的下界，正是它讓過小或過於狹長的參考影像在切 '
      + 'patch 前被放大。未設定 => 本模型族自己的 65536'],
  'cfg.boogu-image-model-config.vl_max_pixels': ['',
      '接地编码：图像处理器的上界。未设置 => 本模型族自己的 16777216',
      '接地編碼：影像處理器的上界。未設定 => 本模型族自己的 16777216'],
  'port.boogu-image-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.boogu-image-model-config.model_config': ['',
      'boogu-image 参数，作为一个 FlexData 对象 {model_family: boogu-image,'
      + ' +vl_*}，供 diffusion-conditioner 的 model_config 输入端口使用（接'
      + '地编码）',
      'boogu-image 參數，作為一個 FlexData 物件 {model_family: boogu-image,'
      + ' +vl_*}，供 diffusion-conditioner 的 model_config 輸入埠使用（接地'
      + '編碼）'],

  // ---- FLUX.2 Model Config (model-specific-config) ----
  'stage.flux2-model-config.name': ['',
      'FLUX.2 模型配置',
      'FLUX.2 模型設定'],
  'stage.flux2-model-config.doc': ['',
      '源：FLUX.2 专属参数——目前主要是声明这是一个 klein-9b-kv 检查点，它那'
      + '套参考图隔离的做法在磁盘上看不出任何痕迹，而一旦搞错方向就会生成错'
      + '误的图像。发一个节拍即结束；若接了 trigger 输入端口，则每收到一个'
      + '节拍发一次。',
      '來源：FLUX.2 專屬參數——目前主要是宣告這是一個 klein-9b-kv 檢查點，它'
      + '那套參考影像隔離的做法在磁碟上看不出任何痕跡，而一旦搞錯方向就會產'
      + '生錯誤的影像。發一個節拍即結束；若接了 trigger 輸入埠，則每收到一'
      + '個節拍發一次。'],
  'cfg.flux2-model-config.klein_kv': ['',
      '该检查点是 FLUX.2-klein-9b-kv 而非普通的 klein-9B：参考 token 在注意'
      + '力中被隔离、以固定时间步 0 调制，且其 K/V 在各去噪步之间缓存（BFL '
      + '实测多参考编辑加速 1.21-2.66 倍）。对该检查点必须开启，对其他检查'
      + '点则是错的——两者在磁盘上无法区分（同样的 config.json、同样的张量名'
      + '），因此无法自动检测。默认 false',
      '該檢查點是 FLUX.2-klein-9b-kv 而非普通的 klein-9B：參考 token 在注意'
      + '力中被隔離、以固定時間步 0 調變，且其 K/V 在各去噪步之間快取（BFL '
      + '實測多參考編輯加速 1.21-2.66 倍）。對該檢查點必須開啟，對其他檢查'
      + '點則是錯的——兩者在磁碟上無法區分（同樣的 config.json、同樣的張量名'
      + '），因此無法自動偵測。預設 false'],
  'cfg.flux2-model-config.lora': ['',
      '在运行时生效的 LoRA——每个被适配的投影计算 W x + scale * B (A x)，而'
      + '不是把增量折进权重里；对 4 位基座而言，这就是“保住一个小修正”和“把'
      + '它四舍五入抹掉”的区别。可以是一个已注册的模型、一个含单个 '
      + '.safetensors 的目录，或指向该文件的路径。模块名必须采用检查点自身'
      + '使用的 diffusers 写法（transformer_blocks.N.attn.to_q、'
      + 'single_transformer_blocks.N.attn.to_qkv_mlp_proj 等），可以带 `'
      + 'diffusion_model.` 前缀；重命名了模块的 BFL/ComfyUI 重打包版本会一'
      + '个都绑不上，并会明确报告。加载时生效：作用在融合投影上的适配器会关'
      + '闭融合 SwiGLU 的交织，因此它在 DiT 构建之前读取，之后再改的节拍只'
      + '会被报告并忽略',
      '在執行階段生效的 LoRA——每個被套用的投影計算 W x + scale * B (A x)，'
      + '而不是把增量摺進權重裡；對 4 位元基座而言，這就是「保住一個小修正'
      + '」和「把它四捨五入抹掉」的差別。可以是一個已註冊的模型、一個含單個'
      + ' .safetensors 的目錄，或指向該檔案的路徑。模組名必須採用檢查點自身'
      + '使用的 diffusers 寫法（transformer_blocks.N.attn.to_q、'
      + 'single_transformer_blocks.N.attn.to_qkv_mlp_proj 等），可以帶 `'
      + 'diffusion_model.` 前綴；重新命名了模組的 BFL/ComfyUI 重新打包版本'
      + '會一個都綁不上，並會明確報告。載入時生效：作用在融合投影上的轉接器'
      + '會關閉融合 SwiGLU 的交織，因此它在 DiT 建構之前讀取，之後再改的節'
      + '拍只會被報告並忽略'],
  'cfg.flux2-model-config.lora_scale': ['',
      '适配器强度，按每次前向生效。可实时调整：它作为常数随 GEMM 一起计算，'
      + '因此无需重新加载即可扫参。1.0 = 与训练时一致；0 会跳过两个适配器 '
      + 'GEMM，所以关掉就是真的关掉',
      '轉接器強度，按每次前向生效。可即時調整：它作為常數隨 GEMM 一起計算，'
      + '因此無需重新載入即可掃參。1.0 = 與訓練時一致；0 會跳過兩個轉接器 '
      + 'GEMM，所以關掉就是真的關掉'],
  'port.flux2-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.flux2-model-config.model_config': ['',
      'FLUX.2 参数，作为一个 FlexData 对象 {model_family: flux2, klein_kv, '
      + '+lora, +lora_scale}，供 generate-image 的 model_config 输入端口使'
      + '用',
      'FLUX.2 參數，作為一個 FlexData 物件 {model_family: flux2, klein_kv, '
      + '+lora, +lora_scale}，供 generate-image 的 model_config 輸入埠使用'],

  // ---- Krea-2 Model Config (model-specific-config) ----
  'stage.krea2-model-config.name': ['',
      'Krea-2 模型配置',
      'Krea-2 模型設定'],
  'stage.krea2-model-config.doc': ['',
      '源：Krea-2 专属参数——它做接地编辑时对源图像编码所用的分辨率，那是身'
      + '份保持编辑 LoRA 的训练分布，而不是可以随意挑的值。发一个节拍即结束'
      + '；若接了 trigger 输入端口，则每收到一个节拍发一次。',
      '來源：Krea-2 專屬參數——它做接地編輯時對來源影像編碼所用的解析度，那'
      + '是身分保持編輯 LoRA 的訓練分布，而不是可以隨意挑的值。發一個節拍即'
      + '結束；若接了 trigger 輸入埠，則每收到一個節拍發一次。'],
  'cfg.krea2-model-config.vl_long_edge': ['',
      '接地编码：送入视觉塔之前，参考图像最长边的上限。未设置 => 768',
      '接地編碼：送入視覺塔之前，參考影像最長邊的上限。未設定 => 768'],
  'cfg.krea2-model-config.vl_pixel_budget': ['',
      '接地编码：总像素数上限，与 vl_long_edge 一同生效。未设置 => 本模型族'
      + '不设该上限',
      '接地編碼：總像素數上限，與 vl_long_edge 一同生效。未設定 => 本模型族'
      + '不設該上限'],
  'cfg.krea2-model-config.vl_min_pixels': ['',
      '接地编码：图像处理器的下界，正是它让过小或过于狭长的参考图在切 patch'
      + ' 前被放大。未设置 => 视觉塔的默认值',
      '接地編碼：影像處理器的下界，正是它讓過小或過於狹長的參考影像在切 '
      + 'patch 前被放大。未設定 => 視覺塔的預設值'],
  'cfg.krea2-model-config.lora': ['',
      '在运行时生效的 LoRA——每个被适配的投影计算 W x + scale * B (A x)，而'
      + '不是把增量折进权重里；对 4 位基座而言，这就是“保住一个小修正”和“把'
      + '它四舍五入抹掉”的区别。可以是一个已注册的模型、一个含单个 '
      + '.safetensors 的目录，或指向该文件的路径。两套键名约定都能绑定：本'
      + '模型自己的 diffusers 命名，以及 ai-toolkit / ComfyUI 的写法。加载'
      + '时生效：它改变权重的构建方式（被适配的 ff.gate/ff.up 不允许融合 '
      + 'SwiGLU 的交织），因此 DiT 起来之后再改的节拍只会被报告并忽略。未设'
      + '置 => 不加适配器',
      '在執行階段生效的 LoRA——每個被套用的投影計算 W x + scale * B (A x)，'
      + '而不是把增量摺進權重裡；對 4 位元基座而言，這就是「保住一個小修正'
      + '」和「把它四捨五入抹掉」的差別。可以是一個已註冊的模型、一個含單個'
      + ' .safetensors 的目錄，或指向該檔案的路徑。兩套鍵名約定都能繫結：本'
      + '模型自己的 diffusers 命名，以及 ai-toolkit / ComfyUI 的寫法。載入'
      + '時生效：它改變權重的建構方式（被套用的 ff.gate/ff.up 不允許融合 '
      + 'SwiGLU 的交織），因此 DiT 起來之後再改的節拍只會被報告並忽略。未設'
      + '定 => 不加轉接器'],
  'cfg.krea2-model-config.lora_scale': ['',
      '适配器强度，按每次前向生效。可实时调整：它作为常数随 GEMM 一起计算，'
      + '因此无需重新加载即可扫参。1.0 = 与训练时一致；0 会跳过两个适配器 '
      + 'GEMM，所以关掉就是真的关掉',
      '轉接器強度，按每次前向生效。可即時調整：它作為常數隨 GEMM 一起計算，'
      + '因此無需重新載入即可掃參。1.0 = 與訓練時一致；0 會跳過兩個轉接器 '
      + 'GEMM，所以關掉就是真的關掉'],
  'cfg.krea2-model-config.vl_max_pixels': ['',
      '接地编码：图像处理器的上界。未设置 => 视觉塔的默认值',
      '接地編碼：影像處理器的上界。未設定 => 視覺塔的預設值'],
  'port.krea2-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.krea2-model-config.model_config': ['',
      'krea2 参数，作为一个 FlexData 对象 {model_family: krea2, +vl_*, +'
      + 'lora, +lora_scale}。其中 vl_* 各键给 diffusion-conditioner 的 '
      + 'model_config 输入端口（接地编码），lora 各键给 generate-image 的。'
      + '请同时接到两者——各自只读取属于自己的部分',
      'krea2 參數，作為一個 FlexData 物件 {model_family: krea2, +vl_*, +'
      + 'lora, +lora_scale}。其中 vl_* 各鍵給 diffusion-conditioner 的 '
      + 'model_config 輸入埠（接地編碼），lora 各鍵給 generate-image 的。請'
      + '同時接到兩者——各自只讀取屬於自己的部分'],

  // ---- Mage-Flow Model Config (model-specific-config) ----
  'stage.mage-flow-model-config.name': ['',
      'Mage-Flow 模型配置',
      'Mage-Flow 模型設定'],
  'stage.mage-flow-model-config.doc': ['',
      '源：Mage-Flow 专属参数——它的来源水印（本模型族独有），以及其参考管道'
      + '对条件图像做接地时所用的分辨率。发一个节拍即结束；若接了 trigger '
      + '输入端口，则每收到一个节拍发一次。',
      '來源：Mage-Flow 專屬參數——它的來源浮水印（本模型族獨有），以及其參考'
      + '管道對條件影像做接地時所用的解析度。發一個節拍即結束；若接了 '
      + 'trigger 輸入埠，則每收到一個節拍發一次。'],
  'cfg.mage-flow-model-config.no_watermark': ['',
      '禁用初始噪声中的 Gaussian-Shading 来源水印。水印默认开启——参考实现无'
      + '条件地施加它，且它保持分布不变，因此不损失画质。这里用否定式命名，'
      + '是为了让安全的默认值不需要任何配置。被固定 init_latents 的运行会忽'
      + '略它',
      '停用初始雜訊中的 Gaussian-Shading 來源浮水印。浮水印預設開啟——參考實'
      + '作無條件地施加它，且它保持分布不變，因此不損失畫質。這裡用否定式命'
      + '名，是為了讓安全的預設值不需要任何設定。被固定 init_latents 的執行'
      + '會忽略它'],
  'cfg.mage-flow-model-config.watermark_key': ['',
      'Gaussian-Shading 密钥：一个整数或一句口令。未设置 => 取 $'
      + 'MAGEFLOW_GS_KEY，否则取 $MAGEFLOW_GS_KEY_FILE / ~/.mageflow/gs_key'
      + '，再否则用公开的默认值。检测端需要使用相同的密钥',
      'Gaussian-Shading 金鑰：一個整數或一句通行語。未設定 => 取 $'
      + 'MAGEFLOW_GS_KEY，否則取 $MAGEFLOW_GS_KEY_FILE / ~/.mageflow/gs_key'
      + '，再否則用公開的預設值。偵測端需要使用相同的金鑰'],
  'cfg.mage-flow-model-config.vl_long_edge': ['',
      '接地编码：送入视觉塔之前，参考图像最长边的上限。未设置 => 本模型族自'
      + '己的 384（pipeline.py 的 vl_cond_long_edge）。调高会偏离接地 LoRA '
      + '的训练分布',
      '接地編碼：送入視覺塔之前，參考影像最長邊的上限。未設定 => 本模型族自'
      + '己的 384（pipeline.py 的 vl_cond_long_edge）。調高會偏離接地 LoRA '
      + '的訓練分布'],
  'cfg.mage-flow-model-config.vl_pixel_budget': ['',
      '接地编码：总像素数上限，与 vl_long_edge 一同生效。未设置 => 本模型族'
      + '不设该上限',
      '接地編碼：總像素數上限，與 vl_long_edge 一同生效。未設定 => 本模型族'
      + '不設該上限'],
  'cfg.mage-flow-model-config.vl_min_pixels': ['',
      '接地编码：图像处理器的下界，正是它让过小或过于狭长的参考图在切 patch'
      + ' 前被放大。未设置 => 本模型族自己的 65536（'
      + 'preprocessor_config.json 的 shortest_edge），远高于 Qwen 默认的 '
      + '3136',
      '接地編碼：影像處理器的下界，正是它讓過小或過於狹長的參考影像在切 '
      + 'patch 前被放大。未設定 => 本模型族自己的 65536（'
      + 'preprocessor_config.json 的 shortest_edge），遠高於 Qwen 預設的 '
      + '3136'],
  'cfg.mage-flow-model-config.vl_max_pixels': ['',
      '接地编码：图像处理器的上界。未设置 => 本模型族自己的 16777216',
      '接地編碼：影像處理器的上界。未設定 => 本模型族自己的 16777216'],
  'port.mage-flow-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.mage-flow-model-config.model_config': ['',
      'Mage-Flow 参数，作为一个 FlexData 对象 {model_family: mage-flow, '
      + 'no_watermark, watermark_key, +vl_*}。请同时接到 generate-image（水'
      + '印）和 diffusion-conditioner（接地编码）——一个检查点，一个配置源',
      'Mage-Flow 參數，作為一個 FlexData 物件 {model_family: mage-flow, '
      + 'no_watermark, watermark_key, +vl_*}。請同時接到 generate-image（浮'
      + '水印）和 diffusion-conditioner（接地編碼）——一個檢查點，一個組態來'
      + '源'],

  // ---- MiniMax-H3 Model Config (model-specific-config) ----
  'stage.minimax-h3-model-config.name': ['',
      'MiniMax-H3 模型配置',
      'MiniMax-H3 模型設定'],
  'stage.minimax-h3-model-config.doc': ['',
      '源：把 MiniMax-H3 专属的生成参数（视频与音频两套噪声计划各自的 sigma'
      + ' shift、固定条件行所处的噪声水平，以及配乐时长）作为一个 FlexData '
      + '节拍发出，供 generate-video 锁存。H3 经过引导蒸馏，因此这里刻意没'
      + '有引导强度。发一个节拍即结束；若接了 trigger 输入端口，则每收到一'
      + '个节拍发一次。',
      '來源：把 MiniMax-H3 專屬的生成參數（影片與音訊兩套雜訊排程各自的 '
      + 'sigma shift、固定條件列所處的雜訊水準，以及配樂長度）作為一個 '
      + 'FlexData 節拍發出，供 generate-video 鎖存。H3 經過引導蒸餾，因此這'
      + '裡刻意沒有引導強度。發一個節拍即結束；若接了 trigger 輸入埠，則每'
      + '收到一個節拍發一次。'],
  'cfg.minimax-h3-model-config.video_shift': ['',
      '视频噪声计划的 sigma shift。12.0 是发布检查点所用的值；它与音频那一'
      + '个不可互换',
      '影片雜訊排程的 sigma shift。12.0 是發布檢查點所用的值；它與音訊那一'
      + '個不可互換'],
  'cfg.minimax-h3-model-config.audio_shift': ['',
      '音频噪声计划的 sigma shift，与视频那一套在相同步数上同步推进',
      '音訊雜訊排程的 sigma shift，與影片那一套在相同步數上同步推進'],
  'cfg.minimax-h3-model-config.condition_timestep': ['',
      '固定关键帧行所处的条件噪声水平；在本模型 t = 1 - sigma 的约定下，1.0'
      + ' 表示完全干净。只有当检查点是用加噪锚点训练的，才需要调低它',
      '固定關鍵影格列所處的條件雜訊水準；在本模型 t = 1 - sigma 的約定下，'
      + '1.0 表示完全乾淨。只有當檢查點是用加噪錨點訓練的，才需要調低它'],
  'cfg.minimax-h3-model-config.condition_audio_timestep': ['',
      '同上，但针对 Ref2VA 参考配乐的那些行。之所以分开，是因为参考的画面和'
      + '音频由不同的 VAE 编码，进入序列时也是不同的行',
      '同上，但針對 Ref2VA 參考配樂的那些列。之所以分開，是因為參考的畫面和'
      + '音訊由不同的 VAE 編碼，進入序列時也是不同的列'],
  'cfg.minimax-h3-model-config.audio_seconds': ['',
      '配乐时长；0 表示由视频的帧数 / 帧率推导，这样两种模态在构造上就等长',
      '配樂長度；0 表示由影片的影格數 / 影格率推導，這樣兩種模態在結構上就'
      + '等長'],
  'cfg.minimax-h3-model-config.lora': ['',
      '在运行时生效的 LoRA——每个被适配的投影计算 W x + scale * B (A x)，而'
      + '不是把增量折进权重里；对 4 位基座而言，这就是“保住一个小修正”和“把'
      + '它四舍五入抹掉”的区别。可以是一个已注册的模型（对 Turbo 适配器执行'
      + ' `model-fetch` 后写入的那个键）、一个含单个 .safetensors 的目录，'
      + '或直接指向该文件的路径。对少步适配器而言，它优于 lora-fuse：'
      + 'MiniMax-H3 Turbo 的增量相对权重只有 2-4e-4，而 bf16 的步进约为 '
      + '4e-3，合并会把其中大部分抹掉（只剩 46-78% 的增量存活），4 位基座则'
      + '更粗。代价约为各投影 FLOPs 的 1.5%。加载时生效：它在 DiT 构建之前'
      + '读取，之后再改的节拍只会被报告并忽略',
      '在執行階段生效的 LoRA——每個被套用的投影計算 W x + scale * B (A x)，'
      + '而不是把增量摺進權重裡；對 4 位元基座而言，這就是「保住一個小修正'
      + '」和「把它四捨五入抹掉」的差別。可以是一個已註冊的模型（對 Turbo '
      + '轉接器執行 `model-fetch` 後寫入的那個鍵）、一個含單個 .safetensors'
      + ' 的目錄，或直接指向該檔案的路徑。對少步轉接器而言，它優於 '
      + 'lora-fuse：MiniMax-H3 Turbo 的增量相對權重只有 2-4e-4，而 bf16 的'
      + '步進約為 4e-3，合併會把其中大部分抹掉（只剩 46-78% 的增量存活），4'
      + ' 位元基座則更粗。代價約為各投影 FLOPs 的 1.5%。載入時生效：它在 '
      + 'DiT 建構之前讀取，之後再改的節拍只會被報告並忽略'],
  'cfg.minimax-h3-model-config.lora_qkv_layout': ['',
      '融合的 attn.qkv_proj 适配器采用哪种行顺序。两家发布方对该投影的行分'
      + '组方式不同——Comfy-Org 的重打包是扁平的 [全部 q | 全部 k | 全部 v]'
      + '，而 MiniMaxAI 的各次发布都是按头排列——本代码树两者都不重排，因此'
      + '为另一方构建的适配器会在每个 block 上落到错误的通道。`auto`（默认'
      + '）按扁平读取（所有已发布的 H3 适配器都是扁平的），并在按头排列的 '
      + 'DiT 上重排行，使一个适配器能同时服务两种检查点。`per_head` 是留给'
      + '真正基于 MiniMaxAI 权重训练的适配器的退路。只影响这个融合投影',
      '融合的 attn.qkv_proj 轉接器採用哪種列順序。兩家發布方對該投影的列分'
      + '組方式不同——Comfy-Org 的重新打包是扁平的 [全部 q | 全部 k | 全部 v'
      + ']，而 MiniMaxAI 的各次發布都是按頭排列——本程式碼樹兩者都不重排，因'
      + '此為另一方建構的轉接器會在每個 block 上落到錯誤的通道。`auto`（預'
      + '設）按扁平讀取（所有已發布的 H3 轉接器都是扁平的），並在按頭排列的'
      + ' DiT 上重排列，使一個轉接器能同時服務兩種檢查點。`per_head` 是留給'
      + '真正基於 MiniMaxAI 權重訓練的轉接器的退路。只影響這個融合投影'],
  'cfg.minimax-h3-model-config.lora_scale': ['',
      '适配器强度，加载时折进 A。1.0 即训练时的强度，也是 Turbo 适配器调校'
      + '的目标值；出现模糊拖影可略微调高，过度锐化起颗粒则调低',
      '轉接器強度，載入時摺進 A。1.0 即訓練時的強度，也是 Turbo 轉接器調校'
      + '的目標值；出現模糊拖影可略微調高，過度銳化起顆粒則調低'],
  'port.minimax-h3-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.minimax-h3-model-config.model_config': ['',
      'MiniMax-H3 生成参数，作为一个 FlexData 对象 {model_family: '
      + 'minimax-h3, video_shift, audio_shift, condition_timestep, '
      + 'condition_audio_timestep, audio_seconds}，供 generate-video 的 '
      + 'model_config 输入端口使用',
      'MiniMax-H3 生成參數，作為一個 FlexData 物件 {model_family: '
      + 'minimax-h3, video_shift, audio_shift, condition_timestep, '
      + 'condition_audio_timestep, audio_seconds}，供 generate-video 的 '
      + 'model_config 輸入埠使用'],

  // ---- Qwen-Image-Edit Model Config (model-specific-config) ----
  'stage.qwen-image-edit-model-config.name': ['',
      'Qwen-Image-Edit 模型配置',
      'Qwen-Image-Edit 模型設定'],
  'stage.qwen-image-edit-model-config.doc': ['',
      '源：Qwen-Image-Edit 专属参数——它的 Qwen2.5-VL 视觉塔据以智能缩放参考'
      + '图的像素预算。发一个节拍即结束；若接了 trigger 输入端口，则每收到'
      + '一个节拍发一次。',
      '來源：Qwen-Image-Edit 專屬參數——它的 Qwen2.5-VL 視覺塔據以智慧縮放參'
      + '考影像的像素預算。發一個節拍即結束；若接了 trigger 輸入埠，則每收'
      + '到一個節拍發一次。'],
  'cfg.qwen-image-edit-model-config.vl_long_edge': ['',
      '接地编码：送入视觉塔之前，参考图像最长边的上限。未设置 => 本模型族不'
      + '设该上限，它的视觉塔改用像素预算',
      '接地編碼：送入視覺塔之前，參考影像最長邊的上限。未設定 => 本模型族不'
      + '設該上限，它的視覺塔改用像素預算'],
  'cfg.qwen-image-edit-model-config.vl_pixel_budget': ['',
      '接地编码：总像素数上限，与 vl_long_edge 一同生效。未设置 => 本模型族'
      + '自己的 384x384（147456）',
      '接地編碼：總像素數上限，與 vl_long_edge 一同生效。未設定 => 本模型族'
      + '自己的 384x384（147456）'],
  'cfg.qwen-image-edit-model-config.vl_min_pixels': ['',
      '接地编码：图像处理器的下界，正是它让过小或过于狭长的参考图在切 patch'
      + ' 前被放大。未设置 => 视觉塔的默认值',
      '接地編碼：影像處理器的下界，正是它讓過小或過於狹長的參考影像在切 '
      + 'patch 前被放大。未設定 => 視覺塔的預設值'],
  'cfg.qwen-image-edit-model-config.vl_max_pixels': ['',
      '接地编码：图像处理器的上界。未设置 => 视觉塔的默认值',
      '接地編碼：影像處理器的上界。未設定 => 視覺塔的預設值'],
  'port.qwen-image-edit-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.qwen-image-edit-model-config.model_config': ['',
      'qwen-image-edit 参数，作为一个 FlexData 对象 {model_family: '
      + 'qwen-image-edit, +vl_*}，供 diffusion-conditioner 的 model_config '
      + '输入端口使用（接地编码）',
      'qwen-image-edit 參數，作為一個 FlexData 物件 {model_family: '
      + 'qwen-image-edit, +vl_*}，供 diffusion-conditioner 的 model_config '
      + '輸入埠使用（接地編碼）'],

  // ---- Wan 2.x Model Config (model-specific-config) ----
  'stage.wan2-model-config.name': ['',
      'Wan 2.x 模型配置',
      'Wan 2.x 模型設定'],
  'stage.wan2-model-config.doc': ['',
      '源：把 Wan 专属的生成参数（无分类器引导，以及 A14B 双专家的切换边界'
      + '）作为一个 FlexData 节拍发出，供 generate-video 锁存。需要这些参数'
      + '的模型族自己携带它们，因此一张图能看出它是为哪个检查点搭的。发一个'
      + '节拍即结束；若接了 trigger 输入端口，则每收到一个节拍发一次。',
      '來源：把 Wan 專屬的生成參數（無分類器引導，以及 A14B 雙專家的切換邊'
      + '界）作為一個 FlexData 節拍發出，供 generate-video 鎖存。需要這些參'
      + '數的模型族自己攜帶它們，因此一張圖能看出它是為哪個檢查點搭的。發一'
      + '個節拍即結束；若接了 trigger 輸入埠，則每收到一個節拍發一次。'],
  'cfg.wan2-model-config.guidance_scale': ['',
      '无分类器引导强度；在双专家检查点（A14B）上指的是高噪声专家的那一个。'
      + 'Wan 未经蒸馏，因此正是它把生成引向提示词——但前提是 generate-video '
      + '上接了负向条件，否则引导强度会被强制为 1',
      '無分類器引導強度；在雙專家檢查點（A14B）上指的是高雜訊專家的那一個。'
      + 'Wan 未經蒸餾，因此正是它把生成引向提示詞——但前提是 generate-video '
      + '上接了負向條件，否則引導強度會被強制為 1'],
  'cfg.wan2-model-config.guidance_scale_2': ['',
      '低噪声专家的无分类器引导强度（双专家检查点，例如 A14B）。与前一个分'
      + '开，是因为两个专家训练所覆盖的 sigma 区间不同',
      '低雜訊專家的無分類器引導強度（雙專家檢查點，例如 A14B）。與前一個分'
      + '開，是因為兩個專家訓練所涵蓋的 sigma 區間不同'],
  'cfg.wan2-model-config.boundary_ratio': ['',
      '低噪声专家从高噪声专家手里接管的 sigma 值；0 表示只有单个专家。请留'
      + '空，让检查点自己的 model_index.json 决定——对所有原版检查点这才是正'
      + '确做法；在这里设置会覆盖该文件',
      '低雜訊專家從高雜訊專家手裡接管的 sigma 值；0 表示只有單個專家。請留'
      + '空，讓檢查點自己的 model_index.json 決定——對所有原版檢查點這才是正'
      + '確做法；在這裡設定會覆寫該檔案'],
  'port.wan2-model-config.trigger': ['',
      '可选的节拍，用于控制何时重新发出配置（chrono 滴答、提示词源、反馈回'
      + '路等）。负载内容不限——收到本身就是信号。未接时，本阶段每次运行只发'
      + '一次',
      '可選的節拍，用於控制何時重新發出設定（chrono 滴答、提示詞來源、回饋'
      + '迴路等）。負載內容不限——收到本身就是訊號。未接時，本階段每次執行只'
      + '發一次'],
  'port.wan2-model-config.model_config': ['',
      'Wan 生成参数，作为一个 FlexData 对象 {model_family: wan, '
      + 'guidance_scale, guidance_scale_2, +boundary_ratio}，供 '
      + 'generate-video 的 model_config 输入端口使用',
      'Wan 生成參數，作為一個 FlexData 物件 {model_family: wan, '
      + 'guidance_scale, guidance_scale_2, +boundary_ratio}，供 '
      + 'generate-video 的 model_config 輸入埠使用'],

  // ---- HLS Broadcast (network) ----
  'stage.hls-broadcast.name': ['', 'HLS 广播', 'HLS 廣播'],
  'stage.hls-broadcast.doc': ['',
      '汇聚节点：把传入的 RGB 帧（VideoToolbox/libx264）和/或 PCM 音频（AAC'
      + '）编码成滚动的内存 HLS 播放列表与分片，并由进程内的 HTTP 服务器对'
      + '外提供。无输出端口。',
      '匯聚節點：把傳入的 RGB 影格（VideoToolbox/libx264）和/或 PCM 音訊（'
      + 'AAC）編碼成滾動的記憶體 HLS 播放清單與分片，並由行程內的 HTTP 伺服'
      + '器對外提供。無輸出埠。'],
  'cfg.hls-broadcast.playlist_name': ['',
      'HLS .m3u8 播放列表文件名',
      'HLS .m3u8 播放清單檔名'],
  'cfg.hls-broadcast.segment_duration': ['',
      'hls_time 秒数（可为小数）',
      'hls_time 秒數（可為小數）'],
  'cfg.hls-broadcast.live_start_offset': ['',
      'EXT-X-START 偏移秒数；0 = 关闭',
      'EXT-X-START 偏移秒數；0 = 關閉'],
  'cfg.hls-broadcast.playlist_max_size': ['',
      'hls_list_size；0 = 不限',
      'hls_list_size；0 = 不限'],
  'cfg.hls-broadcast.hls_flags': ['',
      '封装器的 hls_flags 字符串（\'delete_segments\' 会被忽略——分片存在内存'
      + '中，由内部自行淘汰）',
      '封裝器的 hls_flags 字串（\'delete_segments\' 會被忽略——分片存在記憶體'
      + '中，由內部自行淘汰）'],
  'cfg.hls-broadcast.codec': ['', '视频编码器名称', '視訊編碼器名稱'],
  'cfg.hls-broadcast.bitrate': ['',
      '目标比特率（bps），最小 64000',
      '目標位元率（bps），最小 64000'],
  'cfg.hls-broadcast.gop_size': ['',
      '关键帧间隔（帧数）',
      '關鍵影格間隔（影格數）'],
  'cfg.hls-broadcast.preset': ['', 'libx264 preset', 'libx264 preset'],
  'cfg.hls-broadcast.tune': ['', 'libx264 tune', 'libx264 tune'],
  'cfg.hls-broadcast.fps_num': ['',
      '帧率分子；0 = 自动（沿用输入边带的 fps，否则取 30）',
      '影格率分子；0 = 自動（沿用輸入邊帶的 fps，否則取 30）'],
  'cfg.hls-broadcast.fps_den': ['',
      '帧率分母；0 = 自动（与 fps_num 配对）',
      '影格率分母；0 = 自動（與 fps_num 配對）'],
  'cfg.hls-broadcast.input_normalized': ['',
      'F32 输入是 [0,1] 而非 [0,255]',
      'F32 輸入是 [0,1] 而非 [0,255]'],
  'cfg.hls-broadcast.realtime': ['',
      'PTS 跟随挂钟时间，并强制插入关键帧',
      'PTS 跟隨掛鐘時間，並強制插入關鍵影格'],
  'cfg.hls-broadcast.log_input_stats_every': ['',
      '每第 N 帧记录一次最小/平均/最大值；0 = 关闭',
      '每第 N 格記錄一次最小/平均/最大值；0 = 關閉'],
  'cfg.hls-broadcast.serve_http': ['',
      '运行进程内的静态 HTTP 服务器',
      '執行行程內的靜態 HTTP 伺服器'],
  'cfg.hls-broadcast.bind_address': ['',
      'HTTP 服务器绑定地址；留空 = 自动（本会话由 web 界面提供时用 web 界面'
      + '的地址，否则用 en0 的局域网 IP，再否则 0.0.0.0）',
      'HTTP 伺服器繫結位址；留空 = 自動（本工作階段由 web 介面提供時用 web '
      + '介面的位址，否則用 en0 的區域網路 IP，再否則 0.0.0.0）'],
  'cfg.hls-broadcast.port': ['',
      'HTTP 服务器端口，[0,65535]',
      'HTTP 伺服器連接埠，[0,65535]'],
  'cfg.hls-broadcast.audio_codec': ['',
      '可选音频流所用的音频编码器名称',
      '可選音訊串流所用的音訊編碼器名稱'],
  'cfg.hls-broadcast.audio_bitrate': ['',
      '音频目标比特率（bps）',
      '音訊目標位元率（bps）'],
  'cfg.hls-broadcast.audio_sample_rate': ['',
      '音频编码器输出采样率；0 = 自动（48000）',
      '音訊編碼器輸出取樣率；0 = 自動（48000）'],
  'cfg.hls-broadcast.audio_channels': ['',
      '音频编码器输出声道数',
      '音訊編碼器輸出聲道數'],
  'cfg.hls-broadcast.audio_buffer_seconds': ['',
      '针对没有时间戳的音频所设的初始抖动缓冲（秒）；0 = 关闭',
      '針對沒有時間戳記的音訊所設的初始抖動緩衝（秒）；0 = 關閉'],
  'cfg.hls-broadcast.prime_silence': ['',
      '纯音频实时流：启动时先以一条静音轨上线，让观众能在第一段 PCM 到达之'
      + '前就接入；自计时的节奏会推进它，真实音频接上后无缝续播。默认开启；'
      + '关闭则保持旧行为（第一个音频节拍到达时才出现流）',
      '純音訊即時串流：啟動時先以一條靜音軌上線，讓觀眾能在第一段 PCM 抵達'
      + '之前就接入；自計時的節奏會推進它，真實音訊接上後無縫續播。預設開啟'
      + '；關閉則保持舊行為（第一個音訊節拍抵達時才出現串流）'],
  'port.hls-broadcast.frames': ['',
      '视频 RGB TensorBeat [3,H,W]（F32 或 U8）；分辨率在本阶段的生命周期内'
      + '必须保持不变。若要做纯音频流，把 PCM 生产者接到这里即可（按张量的'
      + '秩自动识别）。',
      '視訊 RGB TensorBeat [3,H,W]（F32 或 U8）；解析度在本階段的生命週期內'
      + '必須保持不變。若要做純音訊串流，把 PCM 生產者接到這裡即可（按張量'
      + '的秩自動辨識）。'],
  'port.hls-broadcast.audio': ['',
      '可选的音频 PCM TensorBeat：F32，秩为 1 的 [n]（单声道）或秩为 2 的 ['
      + '声道数, n]。sideband.sample_rate 与 timestamp_us 会被采用。',
      '可選的音訊 PCM TensorBeat：F32，秩為 1 的 [n]（單聲道）或秩為 2 的 ['
      + '聲道數, n]。sideband.sample_rate 與 timestamp_us 會被採用。'],

  // ---- Preview (network) ----
  'stage.preview.name': ['', '预览', '預覽'],
  'stage.preview.doc': ['',
      '汇聚节点：自计时的实时视频预览（默认 25 fps；没有输入前显示黑屏，之'
      + '后按采样/重复方式出图，并沿用输入的帧率），编码为分片 MP4 的 H.264'
      + '，通过 WebSocket 推送到 web 界面 -> Media Source Extensions（在纯 '
      + 'HTTP 的局域网上即可播放，不需要 HTTPS）。可选地透传 PCM 音频。无输'
      + '出端口。',
      '匯聚節點：自計時的即時視訊預覽（預設 25 fps；沒有輸入前顯示黑畫面，'
      + '之後按取樣/重複方式出圖，並沿用輸入的影格率），編碼為分片 MP4 的 '
      + 'H.264，透過 WebSocket 推送到 web 介面 -> Media Source Extensions（'
      + '在純 HTTP 的區域網路上即可播放，不需要 HTTPS）。可選地透傳 PCM 音'
      + '訊。無輸出埠。'],
  'cfg.preview.bitrate': ['',
      '目标比特率（bps），最小 64000',
      '目標位元率（bps），最小 64000'],
  'cfg.preview.input_normalized': ['',
      'F32 输入是 [0,1] 而非 [0,255]',
      'F32 輸入是 [0,1] 而非 [0,255]'],
  'cfg.preview.title': ['',
      '在 web 界面预览选择器中显示的可选标签；留空 = 使用阶段 id',
      '在 web 介面預覽選擇器中顯示的可選標籤；留空 = 使用階段 id'],
  'cfg.preview.image_mode': ['',
      '对帧率很低的图像类源允许静态图片模式（低于 1 fps 时发送图片而不是视'
      + '频）；false = 始终按视频处理',
      '對影格率很低的影像類來源允許靜態圖片模式（低於 1 fps 時傳送圖片而不'
      + '是視訊）；false = 始終按視訊處理'],
  'port.preview.frames': ['',
      '视频 RGB TensorBeat [3,H,W]（F32 或 U8）；第一帧决定原生分辨率',
      '視訊 RGB TensorBeat [3,H,W]（F32 或 U8）；第一格決定原生解析度'],
  'port.preview.audio': ['',
      '可选的音频 PCM TensorBeat：F32，秩为 1 的 [n]（单声道）或秩为 2 的 ['
      + '声道数, n]。sideband.sample_rate 会被采用。',
      '可選的音訊 PCM TensorBeat：F32，秩為 1 的 [n]（單聲道）或秩為 2 的 ['
      + '聲道數, n]。sideband.sample_rate 會被採用。'],

  // ---- REST Client (network) ----
  'stage.rest-client.name': ['', 'REST 客户端', 'REST 用戶端'],
  'stage.rest-client.doc': ['',
      '每收到一个输入节拍就发起一次 HTTP 请求（libcurl），并把响应（状态码/'
      + '头部/正文）作为 FlexData 信封从 0 号输出端口转发出去。',
      '每收到一個輸入節拍就發起一次 HTTP 請求（libcurl），並把回應（狀態碼/'
      + '標頭/內文）作為 FlexData 信封從 0 號輸出埠轉發出去。'],
  'cfg.rest-client.method': ['',
      'HTTP 方法：GET|POST|PUT|PATCH|DELETE|HEAD',
      'HTTP 方法：GET|POST|PUT|PATCH|DELETE|HEAD'],
  'cfg.rest-client.url': ['',
      '完整的端点 URL（http/https）',
      '完整的端點 URL（http/https）'],
  'cfg.rest-client.headers': ['',
      'header 名 -> 字符串值 的对象',
      '標頭名 -> 字串值 的物件'],
  'cfg.rest-client.payload_path': ['',
      '以斜杠分隔的选择器，指向 0 号输入端口负载中的某处（留空 = 原样使用整'
      + '个负载）',
      '以斜線分隔的選擇器，指向 0 號輸入埠負載中的某處（留空 = 原樣使用整個'
      + '負載）'],
  'cfg.rest-client.payload_format': ['',
      'json | raw_string | none',
      'json | raw_string | none'],
  'cfg.rest-client.content_type': ['',
      '显式覆盖 Content-Type',
      '明確覆寫 Content-Type'],
  'cfg.rest-client.timeout_seconds': ['',
      '网络超时（1..3600）',
      '網路逾時（1..3600）'],
  'cfg.rest-client.verify_tls': ['',
      '强制校验 TLS 证书',
      '強制驗證 TLS 憑證'],
  'cfg.rest-client.emit_on_error': ['',
      '即使发生传输错误也发出一个响应节拍',
      '即使發生傳輸錯誤也發出一個回應節拍'],
  'port.rest-client.request': ['',
      'FlexData 正文来源，或任意触发节拍',
      'FlexData 內文來源，或任意觸發節拍'],
  'port.rest-client.response': ['',
      'FlexData {ok,status,headers,body,body_raw,error,elapsed_ms}',
      'FlexData {ok,status,headers,body,body_raw,error,elapsed_ms}'],

  // ---- RTSP Capture (network) ----
  'stage.rtsp-capture.name': ['', 'RTSP 采集', 'RTSP 擷取'],
  'stage.rtsp-capture.doc': ['',
      '长时间运行的源：连接到已注册摄像头的 RTSP 流（FFmpeg），写出按 IDR '
      + '对齐的 MP4 分段，并按访问单元逐个发出 H.264 与 AAC 的 '
      + 'EncodedSegment。仅 Apple 平台。',
      '長時間執行的來源：連線到已註冊攝影機的 RTSP 串流（FFmpeg），寫出按 '
      + 'IDR 對齊的 MP4 分段，並按存取單元逐個發出 H.264 與 AAC 的 '
      + 'EncodedSegment。僅 Apple 平台。'],
  'cfg.rtsp-capture.camera_name': ['',
      'IP 摄像头注册表中的摄像头键',
      'IP 攝影機註冊表中的攝影機鍵'],
  'cfg.rtsp-capture.output_dir': ['',
      '存放最终 MP4 分段的目录',
      '存放最終 MP4 分段的目錄'],
  'cfg.rtsp-capture.segment_seconds': ['',
      '目标分段长度（秒）',
      '目標分段長度（秒）'],
  'cfg.rtsp-capture.videos_db_suffix': ['',
      '每个摄像头的 videos 子数据库后缀',
      '每個攝影機的 videos 子資料庫後綴'],
  'cfg.rtsp-capture.rtsp_transport': ['',
      'FFmpeg 的 rtsp_transport（tcp/udp）',
      'FFmpeg 的 rtsp_transport（tcp/udp）'],
  'cfg.rtsp-capture.stimeout_us': ['',
      'FFmpeg 套接字 I/O 超时（微秒）',
      'FFmpeg 通訊端 I/O 逾時（微秒）'],
  'cfg.rtsp-capture.connect_timeout_ms': ['',
      '连接超时（毫秒）',
      '連線逾時（毫秒）'],
  'cfg.rtsp-capture.reconnect_delay_ms': ['',
      '重连前的退避时间（毫秒）',
      '重新連線前的退避時間（毫秒）'],
  'cfg.rtsp-capture.rediscover_timeout_ms': ['',
      'ONVIF 重新发现超时（毫秒）',
      'ONVIF 重新探索逾時（毫秒）'],
  'cfg.rtsp-capture.transcode_video_bitrate': ['',
      '需要转码时的 H.264 编码比特率',
      '需要轉碼時的 H.264 編碼位元率'],
  'cfg.rtsp-capture.transcode_audio_bitrate': ['',
      '需要转码时的 AAC 编码比特率',
      '需要轉碼時的 AAC 編碼位元率'],
  'cfg.rtsp-capture.oport_depth': ['',
      '输出环形缓冲深度（丢弃最旧）',
      '輸出環形緩衝深度（丟棄最舊）'],
  'port.rtsp-capture.watchdog': ['',
      '可选的滴答流（例如 chrono）；两次滴答之间摄像头始终没有数据即强制重'
      + '连',
      '可選的滴答串流（例如 chrono）；兩次滴答之間攝影機始終沒有資料即強制'
      + '重新連線'],
  'port.rtsp-capture.video': ['',
      '每个 H.264 访问单元一个 EncodedSegment（AVCC）',
      '每個 H.264 存取單元一個 EncodedSegment（AVCC）'],
  'port.rtsp-capture.audio': ['',
      '每个 AAC 数据包一个 EncodedSegment（不含 ADTS 头）',
      '每個 AAC 封包一個 EncodedSegment（不含 ADTS 標頭）'],

  // ---- LoRA Fuse (preparation) ----
  'stage.lora-fuse.name': ['', 'LoRA 融合', 'LoRA 融合'],
  'stage.lora-fuse.doc': ['',
      '源：把基础模型与 LoRA 适配器融合（W + scale*dW；dW = B@A，LoKr 适配'
      + '器则为 kron(w1,w2)）成一个新的、已注册的模型。支持 diffusers 与 '
      + 'ai-toolkit / ComfyUI（diffusion_model.*）两套适配器命名。对 Krea-2'
      + '，base_model 指向 transformer/ 即 DiT，结果通过 generate-image 的 '
      + 'dit_dir 使用；若同时设置 base_pipeline，还会一并复制编码器/VAE/分'
      + '词器，得到一个自包含、可继续量化的模型。可选的 trigger 输入与 '
      + 'summary 输出。',
      '來源：把基礎模型與 LoRA 轉接器融合（W + scale*dW；dW = B@A，LoKr 轉'
      + '接器則為 kron(w1,w2)）成一個新的、已註冊的模型。支援 diffusers 與 '
      + 'ai-toolkit / ComfyUI（diffusion_model.*）兩套轉接器命名。對 Krea-2'
      + '，base_model 指向 transformer/ 即 DiT，結果透過 generate-image 的 '
      + 'dit_dir 使用；若同時設定 base_pipeline，還會一併複製編碼器/VAE/斷'
      + '詞器，得到一個自包含、可繼續量化的模型。可選的 trigger 輸入與 '
      + 'summary 輸出。'],
  'cfg.lora-fuse.base_model': ['',
      '基础模型目录或 models 数据库键（对 Krea-2 是 transformer/ 即 DiT）',
      '基礎模型目錄或 models 資料庫鍵（對 Krea-2 是 transformer/ 即 DiT）'],
  'cfg.lora-fuse.lora': ['',
      'LoRA .safetensors 文件，或含单个 .safetensors 的目录/数据库键',
      'LoRA .safetensors 檔案，或含單個 .safetensors 的目錄/資料庫鍵'],
  'cfg.lora-fuse.output_name': ['',
      '结果名称 -> <cwd>/models/<output_name>（并注册），或直接给出显式路径',
      '結果名稱 -> <cwd>/models/<output_name>（並註冊），或直接給出明確路徑'],
  'cfg.lora-fuse.base_pipeline': ['',
      '可选的 diffusers 基础管道根目录（目录或数据库键）；设置后，融合出的 '
      + 'DiT 写入 <output>/transformer/，管道的其余组件（text_encoder/、vae'
      + '/、tokenizer/、scheduler/、model_index.json）以硬链接/复制方式并列'
      + '放置 -> 得到一个自包含、可像原版一样继续量化的模型。留空 => 仅输出'
      + '裸 DiT。',
      '可選的 diffusers 基礎管道根目錄（目錄或資料庫鍵）；設定後，融合出的 '
      + 'DiT 寫入 <output>/transformer/，管道的其餘元件（text_encoder/、vae'
      + '/、tokenizer/、scheduler/、model_index.json）以硬連結/複製方式並列'
      + '放置 -> 得到一個自包含、可像原版一樣繼續量化的模型。留空 => 僅輸出'
      + '裸 DiT。'],
  'cfg.lora-fuse.scale': ['',
      'LoRA 融合强度（默认 1.0）',
      'LoRA 融合強度（預設 1.0）'],
  'port.lora-fuse.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，融合会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，融合會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.lora-fuse.summary': ['',
      '融合完成后的 FlexData 摘要；其 `text` 字段可经 save-text 输出为报告'
      + '，该节拍同时触发流程中的下一个阶段',
      '融合完成後的 FlexData 摘要；其 `text` 欄位可經 save-text 輸出為報告'
      + '，該節拍同時觸發流程中的下一個階段'],

  // ---- Model Benchmark (preparation) ----
  'stage.model-benchmark.name': ['', '模型测速', '模型測速'],
  'stage.model-benchmark.doc': ['',
      '源：一次性的语言模型吞吐量基准测试。加载一个语言模型（models 数据库'
      + '键或路径），在若干上下文长度上分别测量预填充、增量预填充与解码的耗'
      + '时，记录每项测试期间的 GPU 温度 / 频率 / 利用率 / 功耗，并输出一份'
      + ' Markdown 报告。可选的 trigger 输入与 summary 输出。',
      '來源：一次性的語言模型吞吐量基準測試。載入一個語言模型（models 資料'
      + '庫鍵或路徑），在若干上下文長度上分別測量預填充、增量預填充與解碼的'
      + '耗時，記錄每項測試期間的 GPU 溫度 / 頻率 / 使用率 / 功耗，並輸出一'
      + '份 Markdown 報告。可選的 trigger 輸入與 summary 輸出。'],
  'cfg.model-benchmark.model': ['',
      '要测速的文本语言模型：models 数据库键（model-fetch / model-quantize'
      + '）或模型目录路径',
      '要測速的文字語言模型：models 資料庫鍵（model-fetch / model-quantize'
      + '）或模型目錄路徑'],
  'cfg.model-benchmark.contexts': ['',
      '以逗号分隔的待测上下文长度（例如再加上 "8192,16384" 做更长的探测）',
      '以逗號分隔的待測上下文長度（例如再加上 "8192,16384" 做更長的探測）'],
  'cfg.model-benchmark.decode_tokens': ['',
      '每次解码测试解码多少 token',
      '每次解碼測試解碼多少 token'],
  'cfg.model-benchmark.prefill_chunk': ['',
      '深度增量预填充探针所用的分块大小',
      '深度增量預填充探針所用的分塊大小'],
  'cfg.model-benchmark.warmup': ['',
      '计时前的热身迭代次数（结果丢弃）',
      '計時前的暖身迭代次數（結果丟棄）'],
  'cfg.model-benchmark.seed': ['',
      '合成提示词的随机种子（同时也是采样器种子）',
      '合成提示詞的隨機種子（同時也是取樣器種子）'],
  'cfg.model-benchmark.cooldown_s': ['',
      '每项计时测试之后休眠多少秒，让 GPU 热状态稳定下来；最后一项之后不再'
      + '休眠',
      '每項計時測試之後休眠多少秒，讓 GPU 熱狀態穩定下來；最後一項之後不再'
      + '休眠'],
  'cfg.model-benchmark.temperature': ['',
      '解码采样器温度（0 = 贪心/argmax）',
      '解碼取樣器溫度（0 = 貪婪/argmax）'],
  'cfg.model-benchmark.top_k': ['',
      '解码采样器的 top-k（0 = 禁用）',
      '解碼取樣器的 top-k（0 = 停用）'],
  'cfg.model-benchmark.top_p': ['',
      '解码采样器的核采样 top-p（1.0 = 禁用）',
      '解碼取樣器的核取樣 top-p（1.0 = 停用）'],
  'cfg.model-benchmark.min_p': ['',
      '解码采样器的 min-p（0.0 = 禁用）',
      '解碼取樣器的 min-p（0.0 = 停用）'],
  'cfg.model-benchmark.repetition_penalty': ['',
      '解码的重复惩罚（1.0 = 禁用）',
      '解碼的重複懲罰（1.0 = 停用）'],
  'cfg.model-benchmark.presence_penalty': ['',
      '解码的存在惩罚（0.0 = 禁用）',
      '解碼的存在懲罰（0.0 = 停用）'],
  'cfg.model-benchmark.mtp': ['',
      '模型带 MTP 头时，解码测试使用 MTP 推测解码（真实文本提示词 + 接受率'
      + '）；否则使用 pdecode 前瞻',
      '模型帶 MTP 頭時，解碼測試使用 MTP 推測解碼（真實文字提示詞 + 接受率'
      + '）；否則使用 pdecode 前瞻'],
  'cfg.model-benchmark.mtp_require_exact': ['',
      '要求 MTP 的贪心解码与串行贪心 token 精确一致后才使用它；不一致时回退'
      + '到 pdecode。默认 false：即使出现分歧，MTP 仍是一次有效的贪心解码（'
      + '批量验证在 bf16 下与单行解码并非逐位相同），因此照样纳入测速并加以'
      + '注明',
      '要求 MTP 的貪婪解碼與序列貪婪 token 精確一致後才使用它；不一致時回退'
      + '到 pdecode。預設 false：即使出現分歧，MTP 仍是一次有效的貪婪解碼（'
      + '批次驗證在 bf16 下與單列解碼並非逐位相同），因此照樣納入測速並加以'
      + '註明'],
  'port.model-benchmark.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，任务会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，工作會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.model-benchmark.summary': ['',
      '本次测速的 FlexData 摘要；其 `text` 字段就是 Markdown 报告（可接一个'
      + ' save-text），该节拍同时触发流程中的下一个阶段',
      '本次測速的 FlexData 摘要；其 `text` 欄位就是 Markdown 報告（可接一個'
      + ' save-text），該節拍同時觸發流程中的下一個階段'],

  // ---- Model Eval (preparation) ----
  'stage.model-eval.name': ['', '模型评估', '模型評估'],
  'stage.model-eval.doc': ['',
      '源：一次性离线评估一个或两个语言模型（models 数据库键或路径）。跑 '
      + 'WikiText-2 困惑度 + 随机抽样的 ARC-Challenge 准确率探针，并记录一'
      + '份 Markdown 报告（给两个模型时是对比报告）。可选的 trigger 输入与 '
      + 'summary 输出。',
      '來源：一次性離線評估一個或兩個語言模型（models 資料庫鍵或路徑）。跑 '
      + 'WikiText-2 困惑度 + 隨機抽樣的 ARC-Challenge 準確率探針，並記錄一'
      + '份 Markdown 報告（給兩個模型時是對比報告）。可選的 trigger 輸入與 '
      + 'summary 輸出。'],
  'cfg.model-eval.model_a': ['',
      '第一个模型：models 数据库键（model-fetch / model-quantize）或模型目'
      + '录路径',
      '第一個模型：models 資料庫鍵（model-fetch / model-quantize）或模型目'
      + '錄路徑'],
  'cfg.model-eval.model_b': ['',
      '可选的第二个模型（键或路径）；设置后报告会是带差值的并排对比',
      '可選的第二個模型（鍵或路徑）；設定後報告會是帶差值的並排對比'],
  'cfg.model-eval.wikitext': ['',
      'WikiText-2 数据集：models 数据库键（可用 model-fetch 获取）或目录路'
      + '径；留空/找不到则跳过困惑度探针',
      'WikiText-2 資料集：models 資料庫鍵（可用 model-fetch 取得）或目錄路'
      + '徑；留空/找不到則跳過困惑度探針'],
  'cfg.model-eval.arc': ['',
      'ARC-Challenge 数据集：models 数据库键或目录路径；留空/找不到则跳过 '
      + 'ARC 探针',
      'ARC-Challenge 資料集：models 資料庫鍵或目錄路徑；留空/找不到則跳過 '
      + 'ARC 探針'],
  'cfg.model-eval.ppl_tokens': ['',
      '用于计算困惑度的 WikiText-2 token 数（0 => 全部）',
      '用於計算困惑度的 WikiText-2 token 數（0 => 全部）'],
  'cfg.model-eval.arc_samples': ['',
      '随机抽取多少道 ARC-Challenge 题目（0 => 整个题库）',
      '隨機抽取多少道 ARC-Challenge 題目（0 => 整個題庫）'],
  'cfg.model-eval.seed': ['',
      'ARC-Challenge 抽样的随机种子',
      'ARC-Challenge 抽樣的隨機種子'],
  'cfg.model-eval.divergence': ['',
      'A 与 B 输出之间的 KL 散度 + logit 相对误差（用于绝对困惑度没有意义的'
      + '模型，例如 MOSS-TTS-Local-v1.5 这类 TTS/编解码骨干）；需要 model_b'
      + '。A 的逐 token logits 会被缓存（ppl_tokens * 词表大小 个浮点数，在'
      + '词表 152k / 256 token 时约 155 MB）——词表极大的模型请调低 '
      + 'ppl_tokens。',
      'A 與 B 輸出之間的 KL 散度 + logit 相對誤差（用於絕對困惑度沒有意義的'
      + '模型，例如 MOSS-TTS-Local-v1.5 這類 TTS/編解碼骨幹）；需要 model_b'
      + '。A 的逐 token logits 會被快取（ppl_tokens * 詞表大小 個浮點數，在'
      + '詞表 152k / 256 token 時約 155 MB）——詞表極大的模型請調低 '
      + 'ppl_tokens。'],
  'port.model-eval.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，任务会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，工作會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.model-eval.summary': ['',
      '本次评估的 FlexData 摘要；其 `text` 字段就是 Markdown 报告（可接一个'
      + ' save-text），该节拍同时触发流程中的下一个阶段',
      '本次評估的 FlexData 摘要；其 `text` 欄位就是 Markdown 報告（可接一個'
      + ' save-text），該節拍同時觸發流程中的下一個階段'],

  // ---- Model Fetch (preparation) ----
  'stage.model-fetch.name': ['', '模型下载', '模型下載'],
  'stage.model-fetch.doc': ['',
      '交互式一次性任务：确定一个模型（浏览内置的 HuggingFace 目录，或直接'
      + '输入路径），从 HuggingFace 或镜像（modelscope.cn）下载到某个基准路'
      + '径下，原生合成 Qwen3-ASR 的 tokenizer.json，并以 huggingface.co 上'
      + '的路径为键注册进模型注册表（无论实际由哪个源提供）。可选的 trigger'
      + ' 输入与 summary 输出。',
      '互動式一次性工作：確定一個模型（瀏覽內建的 HuggingFace 目錄，或直接'
      + '輸入路徑），從 HuggingFace 或鏡像（modelscope.cn）下載到某個基準路'
      + '徑下，原生合成 Qwen3-ASR 的 tokenizer.json，並以 huggingface.co 上'
      + '的路徑為鍵註冊進模型註冊表（無論實際由哪個來源提供）。可選的 '
      + 'trigger 輸入與 summary 輸出。'],
  'cfg.model-fetch.base_path': ['',
      '下载根目录；留空 -> ./models。文件落在 <base>/<owner>/<repo> 下',
      '下載根目錄；留空 -> ./models。檔案落在 <base>/<owner>/<repo> 下'],
  'cfg.model-fetch.model_path': ['',
      '非交互方式：直接给出 \'owner/repo\'（或完整 URL）；留空 -> 交互提示 / '
      + '浏览目录',
      '非互動方式：直接給出 \'owner/repo\'（或完整 URL）；留空 -> 互動提示 / '
      + '瀏覽目錄'],
  'cfg.model-fetch.model_key': ['',
      '在 models 数据库中改用这个键注册，而不用目录名 / hf 路径。这样，同一'
      + '个仓库发布的两个模型可以在磁盘上共用一个目录，同时在数据库中保持为'
      + '两条独立记录：把 Comfy-Org/MiniMax-H3 取两次，一次用 model_variant'
      + '=fl2va + model_key=Comfy-Org/MiniMax-H3-FL2VA，一次用 ref2va，两条'
      + '记录各有自己的 model_type、版本和文件清单，而字节只下载一次（'
      + 'skip_existing_files 会避免共享的编码器和各 VAE 被重复下载）。留空 '
      + '-> 与此前一样，用目录名，否则用 hf 路径',
      '在 models 資料庫中改用這個鍵註冊，而不用目錄名 / hf 路徑。這樣，同一'
      + '個儲存庫發布的兩個模型可以在磁碟上共用一個目錄，同時在資料庫中保持'
      + '為兩筆獨立記錄：把 Comfy-Org/MiniMax-H3 取兩次，一次用 '
      + 'model_variant=fl2va + model_key=Comfy-Org/MiniMax-H3-FL2VA，一次用'
      + ' ref2va，兩筆記錄各有自己的 model_type、版本和檔案清單，而位元組只'
      + '下載一次（skip_existing_files 會避免共用的編碼器和各 VAE 被重複下'
      + '載）。留空 -> 與此前一樣，用目錄名，否則用 hf 路徑'],
  'cfg.model-fetch.model_variant': ['',
      '当一个仓库发布了多个模型、而 `model_path` 无法区分时，指明要哪一个。'
      + '它会（不区分大小写地）与目录条目的 version、variant、name 和 '
      + 'model_type 逐一匹配；任一项的精确命中优先于子串匹配。这是浏览流程'
      + '中版本/变体菜单的非交互写法——例如 Comfy-Org/MiniMax-H3 同时发布了 '
      + 'FL2VA 与 Ref2VA 两个分区，两者固定的文件并不相同，写 "ref2va" 就是'
      + '要第二个。仓库只发布一个模型时留空即可；发布了多个而留空时，取回会'
      + '被拒绝并列出候选，而不是悄悄取第一个',
      '當一個儲存庫發布了多個模型、而 `model_path` 無法區分時，指明要哪一個'
      + '。它會（不區分大小寫地）與目錄條目的 version、variant、name 和 '
      + 'model_type 逐一比對；任一項的精確命中優先於子字串比對。這是瀏覽流'
      + '程中版本/變體選單的非互動寫法——例如 Comfy-Org/MiniMax-H3 同時發布'
      + '了 FL2VA 與 Ref2VA 兩個分區，兩者固定的檔案並不相同，寫 "ref2va" '
      + '就是要第二個。儲存庫只發布一個模型時留空即可；發布了多個而留空時，'
      + '取回會被拒絕並列出候選，而不是悄悄取第一個'],
  'cfg.model-fetch.hf_token': ['',
      '访问受限/私有仓库所用的 bearer token；留空 -> 取所选源对应的环境变量'
      + '（HuggingFace 用 $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN，ModelScope '
      + '用 $MODELSCOPE_API_TOKEN / $MODELSCOPE_TOKEN）；若下载时发现是受限'
      + '仓库，会通过 getpasswd 提示输入',
      '存取受限/私有儲存庫所用的 bearer token；留空 -> 取所選來源對應的環境'
      + '變數（HuggingFace 用 $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN，'
      + 'ModelScope 用 $MODELSCOPE_API_TOKEN / $MODELSCOPE_TOKEN）；若下載'
      + '時發現是受限儲存庫，會透過 getpasswd 提示輸入'],
  'cfg.model-fetch.source': ['',
      '从哪里下载：\'huggingface\'（默认）或 \'modelscope\'（modelscope.cn，中'
      + '国大陆可访问的镜像）。留空 -> 取 $VPIPE_MODEL_SOURCE，否则用 '
      + 'huggingface——这样受网络限制的机器只需设置一次环境变量，而不必逐个'
      + '阶段配置。它只决定下载源：注册表的键和磁盘上的目录一律取自 '
      + 'HuggingFace 路径，因此同一个模型无论从哪一侧取回，都是同一个位置上'
      + '的同一个模型，切换源也不会产生第二份副本。目录中的绝大多数条目在镜'
      + '像上使用完全相同的 owner/repo；少数例外的映射写在 model-source.cc '
      + '中',
      '從哪裡下載：\'huggingface\'（預設）或 \'modelscope\'（modelscope.cn，中'
      + '國大陸可存取的鏡像）。留空 -> 取 $VPIPE_MODEL_SOURCE，否則用 '
      + 'huggingface——這樣受網路限制的機器只需設定一次環境變數，而不必逐個'
      + '階段設定。它只決定下載來源：註冊表的鍵和磁碟上的目錄一律取自 '
      + 'HuggingFace 路徑，因此同一個模型無論從哪一側取回，都是同一個位置上'
      + '的同一個模型，切換來源也不會產生第二份副本。目錄中的絕大多數條目在'
      + '鏡像上使用完全相同的 owner/repo；少數例外的對應寫在 '
      + 'model-source.cc 中'],
  'cfg.model-fetch.source_revision': ['',
      '要读取的分支/标签/提交；留空 -> 各源自己的默认值，而两者并不相同（'
      + 'HuggingFace 是 \'main\'，ModelScope 是 \'master\'）。正因如此，显式指'
      + '定往往并不合适——而且在 ModelScope 上，一个不存在的修订版返回的是空'
      + '文件列表而不是错误',
      '要讀取的分支/標籤/提交；留空 -> 各來源自己的預設值，而兩者並不相同（'
      + 'HuggingFace 是 \'main\'，ModelScope 是 \'master\'）。正因如此，明確指'
      + '定往往並不合適——而且在 ModelScope 上，一個不存在的修訂版傳回的是空'
      + '檔案清單而不是錯誤'],
  'cfg.model-fetch.overwrite_existing': ['',
      '对注册表中已有的模型重新下载并重新注册',
      '對註冊表中已有的模型重新下載並重新註冊'],
  'cfg.model-fetch.prepare_tokenizer': ['',
      '为 Qwen3-ASR 原生合成 tokenizer.json（不需要 transformers）',
      '為 Qwen3-ASR 原生合成 tokenizer.json（不需要 transformers）'],
  'cfg.model-fetch.skip_existing_files': ['',
      '跳过磁盘上已存在且大小与仓库一致的文件',
      '跳過磁碟上已存在且大小與儲存庫一致的檔案'],
  'cfg.model-fetch.verify_tls': ['',
      '强制校验 TLS 证书',
      '強制驗證 TLS 憑證'],
  'cfg.model-fetch.timeout_seconds': ['',
      '元数据调用（列出仓库文件）的截止时间。文件传输改由 stall_seconds 约'
      + '束：总时限无法区分慢连接和死连接，而一个 20 GB 的分片以 2 MB/s 下'
      + '载要三个小时，那完全是健康的',
      '中繼資料呼叫（列出儲存庫檔案）的截止時間。檔案傳輸改由 stall_seconds'
      + ' 約束：總時限無法區分慢連線和死連線，而一個 20 GB 的分片以 2 MB/s '
      + '下載要三個小時，那完全是健康的'],
  'cfg.model-fetch.stall_seconds': ['',
      '文件传输低于 1 KB/s 持续这么久后放弃，并从中断处重试。这才是大分片真'
      + '正需要的超时——它只在连接已经死掉时触发，不会因为单纯的慢而触发。0 '
      + '-> 不做停滞检测（此时挂起的传输会一直阻塞，直到对端断开）',
      '檔案傳輸低於 1 KB/s 持續這麼久後放棄，並從中斷處重試。這才是大分片真'
      + '正需要的逾時——它只在連線已經死掉時觸發，不會因為單純的慢而觸發。0 '
      + '-> 不做停滯偵測（此時掛起的傳輸會一直阻塞，直到對端斷開）'],
  'cfg.model-fetch.download_retries': ['',
      '首次之外每个文件额外尝试的次数，每次都从磁盘上的部分文件继续（间隔 2'
      + '/5/15/30/60 秒）。无论如何部分文件都会在阶段结束后保留，因此用尽次'
      + '数的下载在再次运行时是继续而不是重来',
      '首次之外每個檔案額外嘗試的次數，每次都從磁碟上的部分檔案繼續（間隔 2'
      + '/5/15/30/60 秒）。無論如何部分檔案都會在階段結束後保留，因此用盡次'
      + '數的下載在再次執行時是繼續而不是重來'],
  'cfg.model-fetch.xet_streams': ['',
      '对仓库发布了内容存储哈希的大文件，一次并行拉取多少个区间。这类文件以'
      + '去重、压缩后的分块存储，可以并行取回再重组，而不必用单条连接串行下'
      + '载——单条连接往往在链路带宽之前就先成为瓶颈。在一个 5.2 GB 的 bf16 '
      + '分片上实测，三组交错对比：单流中位 88.3 MB/s，8 路 94.4 MB/s——1.07'
      + ' 倍，其中大部分来自内容存储对 bf16 权重只需传 0.873 倍字节，而非并'
      + '行本身，因为单流本已接近该链路的上限。在单流确为瓶颈的环境下收益要'
      + '大得多。0 -> 一律走普通单流。小于 256 MB 的文件无论如何都走单流：'
      + '为它们多花两个往返并不划算',
      '對儲存庫發布了內容儲存雜湊的大檔案，一次並行拉取多少個區間。這類檔案'
      + '以去重、壓縮後的分塊儲存，可以並行取回再重組，而不必用單條連線序列'
      + '下載——單條連線往往在鏈路頻寬之前就先成為瓶頸。在一個 5.2 GB 的 '
      + 'bf16 分片上實測，三組交錯對比：單串流中位 88.3 MB/s，8 路 94.4 MB/'
      + 's——1.07 倍，其中大部分來自內容儲存對 bf16 權重只需傳 0.873 倍位元'
      + '組，而非並行本身，因為單串流本已接近該鏈路的上限。在單串流確為瓶頸'
      + '的環境下收益要大得多。0 -> 一律走普通單串流。小於 256 MB 的檔案無'
      + '論如何都走單串流：為它們多花兩個往返並不划算'],
  'cfg.model-fetch.verify_checksums': ['',
      '把每个下载下来的文件与其仓库公布的校验和逐一核对——LFS 分片用 SHA-256'
      + '，小文件用 git blob 的 SHA-1——不一致就整个重新下载。HuggingFace 不'
      + '为任何文件公布 MD5，因此没有 MD5 可核。仓库什么都没公布的文件会被'
      + '报告为未校验，而不是算作通过',
      '把每個下載下來的檔案與其儲存庫公布的總和檢查碼逐一核對——LFS 分片用 '
      + 'SHA-256，小檔案用 git blob 的 SHA-1——不一致就整個重新下載。'
      + 'HuggingFace 不為任何檔案公布 MD5，因此沒有 MD5 可核。儲存庫什麼都'
      + '沒公布的檔案會被報告為未檢查，而不是算作通過'],
  'port.model-fetch.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，任务会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，工作會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.model-fetch.summary': ['',
      '已完成工作的 FlexData 摘要；其 `text` 字段可经 save-text 输出为报告'
      + '，该节拍同时触发流程中的下一个阶段',
      '已完成工作的 FlexData 摘要；其 `text` 欄位可經 save-text 輸出為報告'
      + '，該節拍同時觸發流程中的下一個階段'],

  // ---- Model Quantize (preparation) ----
  'stage.model-quantize.name': ['', '模型量化', '模型量化'],
  'stage.model-quantize.doc': ['',
      '源：一次性离线把 safetensors 模型量化为 MLX 仿射分组量化格式，覆盖 '
      + 'vpipe 支持的各个模型族（Llama / Qwen2 / Qwen3 / Qwen3.5 / Gemma-4 '
      + '/ MOSS）。arch / layer_prefix / n_layers 从源自动检测。线性层被量'
      + '化；嵌入/输出头/归一化/辅助模块原样透传。对 Krea-2（文生图）模型，'
      + '`target` 选择组件（dit | text_encoder | vae），输出是一个自包含的'
      + '管道（所有组件都复制过来，只有目标被量化），可直接用作 '
      + 'generate-image 的 hf_dir——多次串联即可量化多个组件。可选的 trigger'
      + ' 输入与 summary 输出。',
      '來源：一次性離線把 safetensors 模型量化為 MLX 仿射分組量化格式，涵蓋'
      + ' vpipe 支援的各個模型族（Llama / Qwen2 / Qwen3 / Qwen3.5 / Gemma-4'
      + ' / MOSS）。arch / layer_prefix / n_layers 從來源自動偵測。線性層被'
      + '量化；嵌入/輸出頭/正規化/輔助模組原樣透傳。對 Krea-2（文生圖）模型'
      + '，`target` 選擇元件（dit | text_encoder | vae），輸出是一個自包含'
      + '的管道（所有元件都複製過來，只有目標被量化），可直接用作 '
      + 'generate-image 的 hf_dir——多次串聯即可量化多個元件。可選的 trigger'
      + ' 輸入與 summary 輸出。'],
  'cfg.model-quantize.src_model': ['',
      '源模型：models 数据库中的键（由 model-fetch 注册），或一个 bf16/f16 '
      + '的 safetensors 目录路径',
      '來源模型：models 資料庫中的鍵（由 model-fetch 註冊），或一個 bf16/'
      + 'f16 的 safetensors 目錄路徑'],
  'cfg.model-quantize.output_name': ['',
      '结果名称 -> <cwd>/models/<output_name>（以该键注册进 models 数据库）'
      + '，或一个显式路径（"/.."、"./.."），后者原样使用且不注册',
      '結果名稱 -> <cwd>/models/<output_name>（以該鍵註冊進 models 資料庫）'
      + '，或一個明確路徑（"/.."、"./.."），後者原樣使用且不註冊'],
  'cfg.model-quantize.bits': ['',
      '骨干网络的仿射量化位宽（4 | 8）',
      '骨幹網路的仿射量化位元寬度（4 | 8）'],
  'cfg.model-quantize.group_size': ['',
      '仿射量化的分组大小（32 | 64）',
      '仿射量化的分組大小（32 | 64）'],
  'cfg.model-quantize.arch': ['',
      '模型族标签；留空 => 从 config.json 的 model_type 自动检测',
      '模型族標籤；留空 => 從 config.json 的 model_type 自動偵測'],
  'cfg.model-quantize.target': ['',
      '量化模型的哪一部分。文生图（Krea-2 / FLUX.2 / Boogu-Image）：选一个'
      + '组件——dit（默认）| text_encoder（Boogu 是它的 mllm/）| vae——输出是'
      + '一个自包含的管道（全部复制，只量化目标），可直接作为 hf_dir 使用；'
      + '可多次串联。通用 / 多模态语言模型：选一个子模块范围——all（默认，整'
      + '个模型）| text（仅语言骨干）| vision | audio | 一个显式的张量名前'
      + '缀——只量化该子模块，其余保持 bf16（例如量化语言骨干、保留全精度视'
      + '觉塔）',
      '量化模型的哪一部分。文生圖（Krea-2 / FLUX.2 / Boogu-Image）：選一個'
      + '元件——dit（預設）| text_encoder（Boogu 是它的 mllm/）| vae——輸出是'
      + '一個自包含的管道（全部複製，只量化目標），可直接作為 hf_dir 使用；'
      + '可多次串聯。通用 / 多模態語言模型：選一個子模組範圍——all（預設，整'
      + '個模型）| text（僅語言骨幹）| vision | audio | 一個明確的張量名前'
      + '綴——只量化該子模組，其餘保持 bf16（例如量化語言骨幹、保留全精度視'
      + '覺塔）'],
  'cfg.model-quantize.skip_existing': ['',
      '若输出的 config.json 已存在则跳过',
      '若輸出的 config.json 已存在則跳過'],
  'cfg.model-quantize.awq': ['',
      'AWQ 激活感知平滑（逐层搜索等效浮点的缩放系数）。适用于标准 layernorm'
      + ' 结构——包括全注意力和 Qwen3.5 的门控 DeltaNet（Llama / Qwen2 / '
      + 'Qwen3 / Qwen3.5 / MOSS）；对 Gemma 的 FFN 归一化和 MoE 则被禁止（'
      + '请改用 mixed 或普通量化）。校准：提供 calib_dir，否则对 Qwen3 系列'
      + '（稠密或 Qwen3.5 混合）会在设备上自动校准',
      'AWQ 啟動感知平滑（逐層搜尋等效浮點的縮放係數）。適用於標準 layernorm'
      + ' 結構——包括全注意力和 Qwen3.5 的閘控 DeltaNet（Llama / Qwen2 / '
      + 'Qwen3 / Qwen3.5 / MOSS）；對 Gemma 的 FFN 正規化和 MoE 則被禁止（'
      + '請改用 mixed 或普通量化）。校準：提供 calib_dir，否則對 Qwen3 系列'
      + '（稠密或 Qwen3.5 混合）會在裝置上自動校準'],
  'cfg.model-quantize.awq_clip': ['',
      '配套的 AWQ 逐组权重裁剪搜索（需要 awq）；默认关闭——在小规模校准集上'
      + '可能反而让末端漂移变差',
      '配套的 AWQ 逐組權重裁剪搜尋（需要 awq）；預設關閉——在小規模校準集上'
      + '可能反而讓末端漂移變差'],
  'cfg.model-quantize.klein_kv': ['',
      'flux2 源是 FLUX.2-klein-9b-kv 而非普通的 klein-9B。它只影响设备上的'
      + '校准，因为校准扫描是以一张参考图为条件的：该检查点会隔离参考 token'
      + '，若按普通的联合注意力去校准，就会针对它根本见不到的分布做裁剪。权'
      + '重量化本身完全相同（张量名与形状一致）。默认 false',
      'flux2 來源是 FLUX.2-klein-9b-kv 而非普通的 klein-9B。它只影響裝置上'
      + '的校準，因為校準掃描是以一張參考影像為條件的：該檢查點會隔離參考 '
      + 'token，若按普通的聯合注意力去校準，就會針對它根本見不到的分布做裁'
      + '剪。權重量化本身完全相同（張量名與形狀一致）。預設 false'],
  'cfg.model-quantize.calib_dir': ['',
      '含 calib_{qkv,o,gateup,down}.f32 激活统计量的目录；留空（默认）=> 对'
      + '已知架构在设备上自动校准',
      '含 calib_{qkv,o,gateup,down}.f32 啟動統計量的目錄；留空（預設）=> 對'
      + '已知架構在裝置上自動校準'],
  'cfg.model-quantize.mixed': ['',
      'unsloth 风格的逐层混合精度：把最敏感的 mixed_frac 比例的层提升到 '
      + 'high_bits。适用于任何采用标准命名的模型族（不匹配的层会被跳过）。'
      + '要求 bits=4、high_bits=8、group_size=64',
      'unsloth 風格的逐層混合精度：把最敏感的 mixed_frac 比例的層提升到 '
      + 'high_bits。適用於任何採用標準命名的模型族（不符合的層會被跳過）。'
      + '要求 bits=4、high_bits=8、group_size=64'],
  'cfg.model-quantize.high_bits': ['',
      '混合精度中提升后的位宽（8）',
      '混合精度中提升後的位元寬度（8）'],
  'cfg.model-quantize.mixed_frac': ['',
      '提升到 high_bits 的层所占比例',
      '提升到 high_bits 的層所占比例'],
  'cfg.model-quantize.layer_prefix': ['',
      'awq/mixed 逐层张量所用的架构层根前缀；留空 => 从 config.json 自动检'
      + '测',
      'awq/mixed 逐層張量所用的架構層根前綴；留空 => 從 config.json 自動偵'
      + '測'],
  'cfg.model-quantize.n_layers': ['',
      'awq/mixed 所用的层数；0 => 从 config.json 自动检测',
      'awq/mixed 所用的層數；0 => 從 config.json 自動偵測'],
  'cfg.model-quantize.quant_exclude': ['',
      '以逗号分隔的张量名子串，命中者保持稠密。只要 `target` 指定的是一个整'
      + '体范围就需要它，因为那条规则会收下所有叶子名不是归一化、也不是嵌入'
      + '的二维浮点张量——这对线性层是对的，对其他长得像线性层的东西则是错的'
      + '。有两类会被误伤：调制表（DiT 的 f32 缩放/偏移行是二维的，但它们不'
      + '是矩阵），以及很小的门控投影，在寥寥几行上做逐组仿射量化全是误差、'
      + '毫无节省。这两种都不会表现为加载失败——检查点照样量化、照样加载，然'
      + '后生成错误的结果',
      '以逗號分隔的張量名子字串，命中者保持稠密。只要 `target` 指定的是一個'
      + '整體範圍就需要它，因為那條規則會收下所有葉子名不是正規化、也不是嵌'
      + '入的二維浮點張量——這對線性層是對的，對其他長得像線性層的東西則是錯'
      + '的。有兩類會被誤傷：調變表（DiT 的 f32 縮放/偏移列是二維的，但它們'
      + '不是矩陣），以及很小的閘控投影，在寥寥幾列上做逐組仿射量化全是誤差'
      + '、毫無節省。這兩種都不會表現為載入失敗——檢查點照樣量化、照樣載入，'
      + '然後產生錯誤的結果'],
  'cfg.model-quantize.quant_vision': ['',
      'text_encoder 场景：连检查点的视觉塔（visual.*）也一起量化，而不是原'
      + '样以 bf16 透传。默认关闭，因为视觉塔对精度敏感，而且相对这里量化的'
      + '编码器的语言部分来说体积不大。若某个多模态编码器只用于纯文本，就值'
      + '得开启——MiniMax-H3 抽取的是 Qwen3-VL 骨干，从不运行它的视觉塔，因'
      + '此那约 1.5 GB 的 bf16 完全是白背',
      'text_encoder 情境：連檢查點的視覺塔（visual.*）也一起量化，而不是原'
      + '樣以 bf16 透傳。預設關閉，因為視覺塔對精度敏感，而且相對這裡量化的'
      + '編碼器的語言部分來說體積不大。若某個多模態編碼器只用於純文字，就值'
      + '得開啟——MiniMax-H3 抽取的是 Qwen3-VL 骨幹，從不執行它的視覺塔，因'
      + '此那約 1.5 GB 的 bf16 完全是白背'],
  'cfg.model-quantize.quant_modulation': ['',
      'Qwen-Image-Edit、Boogu-Image、Wan 与 MiniMax-H3 的 DiT：连 AdaLN 调'
      + '制投影也一起量化（QIE 的 *_mod.1；Boogu 的 norm*.linear；H3 的 '
      + 'adaln_proj.linear）。它们是仍保持 bf16 的最大一批权重——QIE 约 13 '
      + 'GB -> 约 3.4 GB，Boogu 2.1 GB -> 约 1.1 GB；在 H3 上调制部分占 33B'
      + ' 中的 13B，把它留作 bf16 会让 4 位检查点停在约 36 GB——正因如此整个'
      + ' DiT 才塞得进 16 GB 的机器。但它们对精度敏感，所以默认保持 bf16；'
      + '在这里显式开启后会强制为 8 位（而不是主体的位宽）',
      'Qwen-Image-Edit、Boogu-Image、Wan 與 MiniMax-H3 的 DiT：連 AdaLN 調'
      + '變投影也一起量化（QIE 的 *_mod.1；Boogu 的 norm*.linear；H3 的 '
      + 'adaln_proj.linear）。它們是仍保持 bf16 的最大一批權重——QIE 約 13 '
      + 'GB -> 約 3.4 GB，Boogu 2.1 GB -> 約 1.1 GB；在 H3 上調變部分占 33B'
      + ' 中的 13B，把它留作 bf16 會讓 4 位元檢查點停在約 36 GB——正因如此整'
      + '個 DiT 才塞得進 16 GB 的機器。但它們對精度敏感，所以預設保持 bf16'
      + '；在這裡明確開啟後會強制為 8 位元（而不是主體的位元寬度）'],
  'port.model-quantize.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，任务会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，工作會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.model-quantize.summary': ['',
      '已完成工作的 FlexData 摘要；其 `text` 字段可经 save-text 输出为报告'
      + '，该节拍同时触发流程中的下一个阶段',
      '已完成工作的 FlexData 摘要；其 `text` 欄位可經 save-text 輸出為報告'
      + '，該節拍同時觸發流程中的下一個階段'],

  // ---- Model Register (preparation) ----
  'stage.model-register.name': ['', '模型注册', '模型註冊'],
  'stage.model-register.doc': ['',
      '一次性任务：把磁盘上已有的模型目录注册进模型注册表（相当于不下载的 '
      + 'model-fetch）。运行时类型与输入/输出模态都从该目录检测得出——目录匹'
      + '配、config.json、diffusers transformer、GGUF 或 CoreML 包——因此通'
      + '常只需配置一个路径。可选的 trigger 输入与 summary 输出。',
      '一次性工作：把磁碟上已有的模型目錄註冊進模型註冊表（相當於不下載的 '
      + 'model-fetch）。執行階段型別與輸入/輸出模態都從該目錄偵測得出——目錄'
      + '比對、config.json、diffusers transformer、GGUF 或 CoreML 套件——因'
      + '此通常只需設定一個路徑。可選的 trigger 輸入與 summary 輸出。'],
  'cfg.model-register.model_dir': ['',
      '磁盘上要注册的模型目录（也可以是单个 .mlpackage / .gguf 文件）。关于'
      + '它记录的一切——运行时类型、输入/输出模态、尺寸标签——都从该目录检测'
      + '得出',
      '磁碟上要註冊的模型目錄（也可以是單個 .mlpackage / .gguf 檔案）。關於'
      + '它記錄的一切——執行階段型別、輸入/輸出模態、尺寸標籤——都從該目錄偵'
      + '測得出'],
  'cfg.model-register.key': ['',
      '注册所用的键；留空 => 取路径末两段拼成的 "<owner>/<repo>"（即 '
      + 'model-fetch 写入的布局），否则取目录名',
      '註冊所用的鍵；留空 => 取路徑末兩段拼成的 "<owner>/<repo>"（即 '
      + 'model-fetch 寫入的版面），否則取目錄名'],
  'cfg.model-register.model_type': ['',
      '覆盖检测出的运行时提示（例如 qwen3.5、gemma4、flux2、krea2）；留空 ='
      + '> 从目录检测，无法确定时保持为空',
      '覆寫偵測出的執行階段提示（例如 qwen3.5、gemma4、flux2、krea2）；留空'
      + ' => 從目錄偵測，無法確定時保持為空'],
  'cfg.model-register.overwrite_existing': ['',
      '允许这个键从另一个目录手中被接管（默认关闭：覆盖别的模型的键会被拒绝'
      + '）。对同一个目录重新注册是幂等的，不需要开启此项',
      '允許這個鍵從另一個目錄手中被接管（預設關閉：覆寫別的模型的鍵會被拒絕'
      + '）。對同一個目錄重新註冊是冪等的，不需要開啟此項'],
  'port.model-register.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，任务会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，工作會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.model-register.summary': ['',
      '本次注册的 FlexData 摘要；其 `text` 字段可经 save-text 输出为报告，'
      + '该节拍同时触发流程中的下一个阶段',
      '本次註冊的 FlexData 摘要；其 `text` 欄位可經 save-text 輸出為報告，'
      + '該節拍同時觸發流程中的下一個階段'],

  // ---- Model Remove (preparation) ----
  'stage.model-remove.name': ['', '模型移除', '模型移除'],
  'stage.model-remove.doc': ['',
      '一次性任务：把一个已注册的模型从 models 数据库中移除（model-fetch 的'
      + '逆操作）。删除以 `model` 为键的记录；若 delete_files=true，还会把'
      + '记录中的 local_path 目录从磁盘上删掉。开启 missing_ok 时，模型本就'
      + '不存在也算成功。可选的 trigger 输入与 summary 输出。',
      '一次性工作：把一個已註冊的模型從 models 資料庫中移除（model-fetch 的'
      + '逆操作）。刪除以 `model` 為鍵的記錄；若 delete_files=true，還會把'
      + '記錄中的 local_path 目錄從磁碟上刪掉。開啟 missing_ok 時，模型本就'
      + '不存在也算成功。可選的 trigger 輸入與 summary 輸出。'],
  'cfg.model-remove.model': ['',
      '要移除的 models 数据库键（即 model-fetch / model-quantize 注册该模型'
      + '时所用的键）',
      '要移除的 models 資料庫鍵（即 model-fetch / model-quantize 註冊該模型'
      + '時所用的鍵）'],
  'cfg.model-remove.delete_files': ['',
      '同时把模型记录中的 local_path 目录从磁盘上删除（需显式开启；具破坏性'
      + '——否则只删除记录）',
      '同時把模型記錄中的 local_path 目錄從磁碟上刪除（需明確開啟；具破壞性'
      + '——否則只刪除記錄）'],
  'cfg.model-remove.missing_ok': ['',
      '把“模型未注册”视为成功（等同于已移除），而不是中断整套流程',
      '把「模型未註冊」視為成功（等同於已移除），而不是中斷整套流程'],
  'port.model-remove.trigger': ['',
      '可选的节奏触发（任意节拍类型）；接上后，任务会先等一个节拍再执行——让'
      + '这些准备类阶段可以串成一套流程',
      '可選的節奏觸發（任意節拍類型）；接上後，工作會先等一個節拍再執行——讓'
      + '這些準備類階段可以串成一套流程'],
  'port.model-remove.summary': ['',
      '已完成工作的 FlexData 摘要；其 `text` 字段可经 save-text 输出为报告'
      + '，该节拍同时触发流程中的下一个阶段',
      '已完成工作的 FlexData 摘要；其 `text` 欄位可經 save-text 輸出為報告'
      + '，該節拍同時觸發流程中的下一個階段'],

  // ---- ONVIF Discovery (preparation) ----
  'stage.onvif-discovery.name': ['',
      'ONVIF 摄像头搜索',
      'ONVIF 攝影機搜尋'],
  'stage.onvif-discovery.doc': ['',
      '交互式一次性任务：用 WS-Discovery 探测 ONVIF 摄像头，提示用户选择并'
      + '输入凭据，然后把加密后的摄像头记录持久化到 LMDB。0 入 / 0 出（仅 '
      + 'Apple 平台）。',
      '互動式一次性工作：用 WS-Discovery 探測 ONVIF 攝影機，提示使用者選擇'
      + '並輸入憑證，然後把加密後的攝影機記錄持久化到 LMDB。0 入 / 0 出（僅'
      + ' Apple 平台）。'],
  'cfg.onvif-discovery.probe_timeout_ms': ['',
      'WS-Discovery 探测超时（毫秒）',
      'WS-Discovery 探測逾時（毫秒）'],
  'cfg.onvif-discovery.overwrite_existing': ['',
      '替换已经注册过的摄像头',
      '取代已經註冊過的攝影機'],
  'cfg.onvif-discovery.mask_password': ['',
      '输入密码时进行遮蔽（getpasswd：关闭标准输入回显 / web 界面使用遮蔽字'
      + '段）；false 则明文显示',
      '輸入密碼時進行遮蔽（getpasswd：關閉標準輸入回顯 / web 介面使用遮蔽欄'
      + '位）；false 則明文顯示'],

  // ---- Load Text (text) ----
  'stage.load-text.name': ['', '加载文本', '載入文字'],
  'stage.load-text.doc': ['',
      '源：从文件系统读取文本文件，把每个文件的内容作为 FlexData 字符串（即'
      + ' text-input 的格式）发出。若接了 trigger 输入端口，则每个节拍发出'
      + '一个文件；未接时一次发完全部然后结束。',
      '來源：從檔案系統讀取文字檔，把每個檔案的內容作為 FlexData 字串（即 '
      + 'text-input 的格式）發出。若接了 trigger 輸入埠，則每個節拍發出一個'
      + '檔案；未接時一次發完全部然後結束。'],
  'cfg.load-text.path': ['',
      '文本文件路径字符串，或字符串数组',
      '文字檔路徑字串，或字串陣列'],
  'port.load-text.trigger': ['',
      '可选的节奏节拍（例如 chrono）；每来一个就读取并发出下一个文件',
      '可選的節奏節拍（例如 chrono）；每來一個就讀取並發出下一個檔案'],
  'port.load-text.text': ['',
      '文件内容，作为一个 FlexData 字符串负载',
      '檔案內容，作為一個 FlexData 字串負載'],

  // ---- Save Text (text) ----
  'stage.save-text.name': ['', '保存文本', '儲存文字'],
  'stage.save-text.doc': ['',
      '汇聚节点：把每个 FlexData 节拍中的某个文本字段（配置项 `key`，默认 "'
      + 'text"）追加写入文本文件，按换行策略每个节拍写一条。EOS 结束本阶段'
      + '。',
      '匯聚節點：把每個 FlexData 節拍中的某個文字欄位（設定項 `key`，預設 "'
      + 'text"）附加寫入文字檔，按換行策略每個節拍寫一條。EOS 結束本階段。'],
  'cfg.save-text.path': ['', '输出文本文件路径', '輸出文字檔路徑'],
  'cfg.save-text.key': ['',
      '要写出的 FlexData 字段（纯字符串负载则整体写出）',
      '要寫出的 FlexData 欄位（純字串負載則整體寫出）'],
  'cfg.save-text.newline': ['',
      '条目分隔策略：after（默认）| before | none',
      '條目分隔策略：after（預設）| before | none'],
  'cfg.save-text.append': ['',
      '追加写入文件（默认）；设为 false 则在开始时清空文件',
      '附加寫入檔案（預設）；設為 false 則在開始時清空檔案'],
  'port.save-text.text': ['',
      '携带待保存文本的 FlexData（见 `key`）',
      '攜帶待儲存文字的 FlexData（見 `key`）'],

  // ---- Chat (text) ----
  'stage.text-chat.name': ['', '聊天', '聊天'],
  'stage.text-chat.doc': ['',
      '对话式语言模型阶段：把每个用户回合追加到持久的 K/V 聊天上下文，把助'
      + '手回复流式送到界面，并发出一条 FlexData 回合记录。/clear 可重置。'
      + '回合中的媒体行标记可附带图像/音频（用 FFmpeg 解码，由模型自己的塔'
      + '编码）；不支持的模态会给出警告并丢弃。',
      '對話式語言模型階段：把每個使用者回合附加到持久的 K/V 聊天上下文，把'
      + '助理回覆串流送到介面，並發出一筆 FlexData 回合記錄。/clear 可重設'
      + '。回合中的媒體行標記可附帶影像/音訊（用 FFmpeg 解碼，由模型自己的'
      + '塔編碼）；不支援的模態會給出警告並丟棄。'],
  'cfg.text-chat.hf_dir': ['',
      '模型：models 数据库中的键（由 model-fetch 注册）或 HF 风格的模型目录'
      + '；同名时数据库键优先于路径。',
      '模型：models 資料庫中的鍵（由 model-fetch 註冊）或 HF 風格的模型目錄'
      + '；同名時資料庫鍵優先於路徑。'],
  'cfg.text-chat.compute_dtype': ['',
      'bf16 | f16 | f32',
      'bf16 | f16 | f32'],
  'cfg.text-chat.page_tokens': ['',
      'ContextManager 的 K/V 分页大小',
      'ContextManager 的 K/V 分頁大小'],
  'cfg.text-chat.max_pages': ['',
      '每个语言模型的分页池容量（>= 1）',
      '每個語言模型的分頁池容量（>= 1）'],
  'cfg.text-chat.max_new_tokens': ['',
      '每个回合的生成预算（>= 1）',
      '每個回合的生成預算（>= 1）'],
  'cfg.text-chat.disable_thinking': ['',
      '覆盖聊天模板的思考（thinking）默认设置',
      '覆寫聊天範本的思考（thinking）預設設定'],
  'cfg.text-chat.reasoning_effort': ['',
      '要求模型思考多少："xhigh"、"medium" 或 "low"。这是 Qwen3.8 聊天模板'
      + '里的控制项——它会在系统回合前加上一段指令（启用工具时并入 tools 回'
      + '合），文本与官方模板逐字一致。"medium" 就是不加指令的渲染结果，正'
      + '如参考实现所定义，因此它什么都不输出。留空（默认）同样什么都不输出'
      + '，这也是此前每个 Qwen3 版本的行为——参考实现自己的默认值是 "xhigh"'
      + '，但 vpipe 无法把 3.8 的检查点与 3.5/3.6 区分开（三者都声明 '
      + 'model_type 为 "qwen3_5"），因此 xhigh 需要显式选择而不是默认假定。'
      + '关闭思考时，或模型族没有推理力度这一概念时，本项会被忽略并给出警告',
      '要求模型思考多少："xhigh"、"medium" 或 "low"。這是 Qwen3.8 聊天範本'
      + '裡的控制項——它會在系統回合前加上一段指令（啟用工具時併入 tools 回'
      + '合），文字與官方範本逐字一致。"medium" 就是不加指令的算繪結果，正'
      + '如參考實作所定義，因此它什麼都不輸出。留空（預設）同樣什麼都不輸出'
      + '，這也是此前每個 Qwen3 版本的行為——參考實作自己的預設值是 "xhigh"'
      + '，但 vpipe 無法把 3.8 的檢查點與 3.5/3.6 區分開（三者都宣告 '
      + 'model_type 為 "qwen3_5"），因此 xhigh 需要明確選擇而不是預設假定。'
      + '關閉思考時，或模型族沒有推理力度這一概念時，本項會被忽略並給出警告'],
  'cfg.text-chat.enable_tools': ['',
      '启用 MCP 工具调用：向模型公布内置工具（get_current_time），执行模型'
      + '发出的任何 <tool_call>，并把结果回灌以进行下一轮解码（仅限 ChatML/'
      + 'Qwen 系列）',
      '啟用 MCP 工具呼叫：向模型公布內建工具（get_current_time），執行模型'
      + '發出的任何 <tool_call>，並把結果回灌以進行下一輪解碼（僅限 ChatML/'
      + 'Qwen 系列）'],
  'cfg.text-chat.enable_python_tool': ['',
      '增加一个沙盒化的 `run_python` 工具（seatbelt：无网络、临时暂存文件系'
      + '统、禁止读取主目录、限制 CPU 与时间）。独立开关；任一工具开关都会'
      + '激活工具循环。它执行的是模型写出的代码——请谨慎开启',
      '增加一個沙箱化的 `run_python` 工具（seatbelt：無網路、暫時暫存檔案系'
      + '統、禁止讀取家目錄、限制 CPU 與時間）。獨立開關；任一工具開關都會'
      + '啟用工具迴圈。它執行的是模型寫出的程式碼——請謹慎開啟'],
  'cfg.text-chat.enable_file_tools': ['',
      '增加受限的 read_file / write_file / list_files 工具，根目录为 '
      + 'file_sandbox_dir（路径无法逃出该根目录）。任一工具开关都会激活工具'
      + '循环',
      '增加受限的 read_file / write_file / list_files 工具，根目錄為 '
      + 'file_sandbox_dir（路徑無法逃出該根目錄）。任一工具開關都會啟用工具'
      + '迴圈'],
  'cfg.text-chat.file_sandbox_dir': ['',
      '文件/shell 工具的显式工作区根目录；设置后覆盖 file_sandbox。留空 => '
      + '使用 file_sandbox 模式',
      '檔案/shell 工具的明確工作區根目錄；設定後覆寫 file_sandbox。留空 => '
      + '使用 file_sandbox 模式'],
  'cfg.text-chat.file_sandbox': ['',
      '未设置 file_sandbox_dir 时，文件/shell 工具的工作区：\'per-chat\'（每'
      + '个阶段一个临时目录，拆除时清空——默认）或 \'persistent\'（会话沙盒根'
      + '目录 $CWD/sandbox，跨会话保留）',
      '未設定 file_sandbox_dir 時，檔案/shell 工具的工作區：\'per-chat\'（每'
      + '個階段一個暫存目錄，拆除時清空——預設）或 \'persistent\'（工作階段沙'
      + '箱根目錄 $CWD/sandbox，跨工作階段保留）'],
  'cfg.text-chat.enable_shell_tool': ['',
      '增加一个沙盒化的 `run_shell` 工具（seatbelt：无网络、写入限制在 '
      + 'file_sandbox_dir 工作区内、禁止读取主目录、限制 CPU 与时间）。与文'
      + '件工具共用工作区。它执行的是模型写出的命令——请谨慎开启',
      '增加一個沙箱化的 `run_shell` 工具（seatbelt：無網路、寫入限制在 '
      + 'file_sandbox_dir 工作區內、禁止讀取家目錄、限制 CPU 與時間）。與檔'
      + '案工具共用工作區。它執行的是模型寫出的指令——請謹慎開啟'],
  'cfg.text-chat.enable_web_tools': ['',
      '增加 fetch_url / scrape_page 工具（http/https GET；带 SSRF 防护：拒'
      + '绝私有/本机目标）。任一工具开关都会激活工具循环',
      '增加 fetch_url / scrape_page 工具（http/https GET；帶 SSRF 防護：拒'
      + '絕私有/本機目標）。任一工具開關都會啟用工具迴圈'],
  'cfg.text-chat.web_allow_private': ['',
      '允许网页工具访问私有/本机地址（关闭 SSRF 防护）——仅限可信的本地使用',
      '允許網頁工具存取私有/本機位址（關閉 SSRF 防護）——僅限可信的本機使用'],
  'cfg.text-chat.allow_system_temp': ['',
      '允许沙盒中的 run_shell / run_python 工具把每用户的系统临时目录（'
      + 'macOS Darwin temp）用作 $TMPDIR 和一个可写根目录。默认 false 会把'
      + '所有临时文件限制在工作区/当前目录之下；开启它是为了让那些硬编码系'
      + '统临时目录、忽略 $TMPDIR 的工具（例如 macOS 的 `mktemp` 命令）能正'
      + '常工作，代价是临时文件会落在启动目录之外',
      '允許沙箱中的 run_shell / run_python 工具把每使用者的系統暫存目錄（'
      + 'macOS Darwin temp）用作 $TMPDIR 和一個可寫根目錄。預設 false 會把'
      + '所有暫存檔限制在工作區/目前目錄之下；開啟它是為了讓那些硬寫死系統'
      + '暫存目錄、忽略 $TMPDIR 的工具（例如 macOS 的 `mktemp` 指令）能正常'
      + '運作，代價是暫存檔會落在啟動目錄之外'],
  'cfg.text-chat.stream_answer_only': ['',
      '把推理（<think>）与工具调用块从流式输出端口（编号 1）中剔除，好让说'
      + '话的消费者（例如 text-to-speech）只朗读答案；0 号输出端口仍然携带'
      + '完整回复',
      '把推理（<think>）與工具呼叫區塊從串流輸出埠（編號 1）中剔除，好讓說'
      + '話的消費者（例如 text-to-speech）只朗讀答案；0 號輸出埠仍然攜帶完'
      + '整回覆'],
  'cfg.text-chat.mtp': ['',
      '模型自带 MTP 推测解码头时使用它（token 精确；只影响性能）；设为 '
      + 'false 则强制走标准解码路径',
      '模型自帶 MTP 推測解碼頭時使用它（token 精確；只影響效能）；設為 '
      + 'false 則強制走標準解碼路徑'],
  'cfg.text-chat.mtp_prefix_seed': ['',
      '解码开始时用提示词为 MTP 起草器的 KV 预热：草稿接受率/解码吞吐更高，'
      + '代价是少量额外预填充（token 精确；只影响性能）。默认开启（聊天是解'
      + '码受限的）；若要偏向预填充吞吐可设为 false。未开启 mtp 时无效。',
      '解碼開始時用提示詞為 MTP 起草器的 KV 預熱：草稿接受率/解碼吞吐更高，'
      + '代價是少量額外預填充（token 精確；只影響效能）。預設開啟（聊天是解'
      + '碼受限的）；若要偏向預填充吞吐可設為 false。未開啟 mtp 時無效。'],
  'cfg.text-chat.i8_prefill': ['',
      '加速模式（有损）：预填充 GEMM 改用动态 int8，在矩阵核心 GPU 上速率约'
      + '为 f16 的 2 倍，精度即 int8 水平（开启后预填充不再 token 精确；解'
      + '码不受影响；没有 NAX matmul2d（矩阵核心 GPU + 相应内核）时会被忽略'
      + '）。默认 false；环境变量 VPIPE_I8_GEMM 可覆盖。',
      '加速模式（失真）：預填充 GEMM 改用動態 int8，在矩陣核心 GPU 上速率約'
      + '為 f16 的 2 倍，精度即 int8 水準（開啟後預填充不再 token 精確；解'
      + '碼不受影響；沒有 NAX matmul2d（矩陣核心 GPU + 相應核心）時會被忽略'
      + '）。預設 false；環境變數 VPIPE_I8_GEMM 可覆寫。'],
  'port.text-chat.user': ['',
      'FlexData 字符串：用户这一回合的文本；可以用媒体行标记内嵌图像/音频附'
      + '件（文件系统路径或 base64），它们会经模型自己的视觉/音频塔拼接进预'
      + '填充',
      'FlexData 字串：使用者這一回合的文字；可以用媒體行標記內嵌影像/音訊附'
      + '件（檔案系統路徑或 base64），它們會經模型自己的視覺/音訊塔拼接進預'
      + '填充'],
  'port.text-chat.sampler': ['',
      '可选的 token 采样器配置 FlexData（sampler-select）。在第一个回合时锁'
      + '存，之后每一回合都沿用；未接线 = 贪心（argmax）解码',
      '可選的 token 取樣器組態 FlexData（sampler-select）。在第一個回合時鎖'
      + '存，之後每一回合都沿用；未接線 = 貪婪（argmax）解碼'],
  'port.text-chat.assistant': ['',
      '每回合的 FlexData {text,prefill_ms,decode_ms,ctx_pos}（下游可选）',
      '每回合的 FlexData {text,prefill_ms,decode_ms,ctx_pos}（下游可選）'],
  'port.text-chat.stream': ['',
      '回复过程中逐步发出的 FlexData {text,end_of_response}——大约每 20 个词'
      + '在句读处（中英文皆可）发一个节拍；一次回复的最后一个节拍会把 '
      + 'end_of_response 置为 true。它供流式消费者（例如 text-to-speech）使'
      + '用，让其在整个回合结束之前就开始工作。开启 stream_answer_only 时，'
      + '推理与工具调用片段会被剔除（下游可选）',
      '回覆過程中逐步發出的 FlexData {text,end_of_response}——大約每 20 個詞'
      + '在句讀處（中英文皆可）發一個節拍；一次回覆的最後一個節拍會把 '
      + 'end_of_response 置為 true。它供串流消費者（例如 text-to-speech）使'
      + '用，讓其在整個回合結束之前就開始工作。開啟 stream_answer_only 時，'
      + '推理與工具呼叫片段會被剔除（下游可選）'],

  // ---- Text Input (text) ----
  'stage.text-input.name': ['', '文本输入', '文字輸入'],
  'stage.text-input.doc': ['',
      '源：在标准输出上给出提示，从标准输入读一行，并把它作为 FlexData 字符'
      + '串发出。若接了 trigger 输入端口，则每个节拍触发一轮新的提示/读取。',
      '來源：在標準輸出上給出提示，從標準輸入讀一行，並把它作為 FlexData 字'
      + '串發出。若接了 trigger 輸入埠，則每個節拍觸發一輪新的提示/讀取。'],
  'cfg.text-input.prompt': ['',
      '每次读取前显示给用户的文本',
      '每次讀取前顯示給使用者的文字'],
  'cfg.text-input.count': ['',
      '读取多少行；0 = 一直读到 EOF',
      '讀取多少行；0 = 一直讀到 EOF'],
  'cfg.text-input.present_first_without_beat': ['',
      '第一次提示不等待输入端口的节拍',
      '第一次提示不等待輸入埠的節拍'],
  'cfg.text-input.media': ['',
      '改用 getmedialine 提示：这一行可以用媒体行标记内嵌图像/音频附件（在'
      + '标准输入上是文件路径，通过 web 界面的附件/拖放控件则是 base64）',
      '改用 getmedialine 提示：這一行可以用媒體行標記內嵌影像/音訊附件（在'
      + '標準輸入上是檔案路徑，透過 web 介面的附件/拖放控制項則是 base64）'],
  'port.text-input.trigger': ['',
      '可选的节拍，用于触发下一轮提示/读取（例如一个反馈回路）',
      '可選的節拍，用於觸發下一輪提示/讀取（例如一個回饋迴路）'],
  'port.text-input.text': ['',
      '每读入一行发出一个 FlexData 字符串负载',
      '每讀入一行發出一個 FlexData 字串負載'],

  // ---- Text Prompt (text) ----
  'stage.text-prompt.name': ['', '文本提示', '文字提示'],
  'stage.text-prompt.doc': ['',
      '源：把配置的 `text` 作为一个纯 FlexData 字符串发出（即 text-input 的'
      + '格式，不带媒体）。发一个节拍即结束；若接了 trigger 输入端口，则每'
      + '收到一个节拍发一次。',
      '來源：把設定的 `text` 作為一個純 FlexData 字串發出（即 text-input 的'
      + '格式，不帶媒體）。發一個節拍即結束；若接了 trigger 輸入埠，則每收'
      + '到一個節拍發一次。'],
  'cfg.text-prompt.text': ['',
      '要作为 FlexData 字符串发出的提示文本',
      '要作為 FlexData 字串發出的提示文字'],
  'port.text-prompt.trigger': ['',
      '可选的节拍，用于控制何时重新发出该文本（例如 chrono 滴答或反馈回路）',
      '可選的節拍，用於控制何時重新發出該文字（例如 chrono 滴答或回饋迴路）'],
  'port.text-prompt.text': ['',
      '配置的文本，作为一个 FlexData 字符串负载',
      '設定的文字，作為一個 FlexData 字串負載'],

  // ---- ByteTrack (vision) ----
  'stage.byte-track.name': ['', '目标跟踪器', '物件追蹤器'],
  'stage.byte-track.doc': ['',
      'ByteTrack 多目标跟踪器：为 yolo 检测结果跨帧分配持久的 track_id（卡'
      + '尔曼滤波 + 两阶段 IoU/匈牙利匹配）。转发已确认的检测结果。',
      'ByteTrack 多物件追蹤器：為 yolo 偵測結果跨影格分配持久的 track_id（'
      + '卡爾曼濾波 + 兩階段 IoU/匈牙利比對）。轉發已確認的偵測結果。'],
  'cfg.byte-track.track_thresh': ['',
      '分数大于等于该值的进入高分池',
      '分數大於等於該值的進入高分池'],
  'cfg.byte-track.high_thresh': ['',
      '未匹配的检测要生成新轨迹所需跨过的门槛',
      '未比對到的偵測要產生新軌跡所需跨過的門檻'],
  'cfg.byte-track.match_thresh': ['',
      '首轮关联的代价上限（1 - IoU）',
      '首輪關聯的成本上限（1 - IoU）'],
  'cfg.byte-track.frame_rate': ['',
      '与 track_buffer 一起决定最多可丢失多少帧',
      '與 track_buffer 一起決定最多可遺失多少影格'],
  'cfg.byte-track.track_buffer': ['',
      '轨迹丢失后等待多少帧才被移除',
      '軌跡遺失後等待多少影格才被移除'],
  'cfg.byte-track.oport_capacity': ['',
      '输出环形缓冲容量',
      '輸出環形緩衝容量'],
  'port.byte-track.detections': ['',
      'FlexData 形式的 yolo-detection 帧记录 {frame_width,frame_height,'
      + 'detections[]}',
      'FlexData 形式的 yolo-detection 影格記錄 {frame_width,frame_height,'
      + 'detections[]}'],
  'port.byte-track.tracks': ['',
      '同样的 FlexData 结构；只含已确认的检测，且每一个都附上了整数 '
      + 'track_id',
      '同樣的 FlexData 結構；只含已確認的偵測，且每一個都附上了整數 '
      + 'track_id'],

  // ---- Detection Overlay (vision) ----
  'stage.detection-overlay.name': ['', '检测结果可视化', '偵測結果渲染器'],
  'stage.detection-overlay.doc': ['',
      '把得分最高的 N 个检测框与标签（以及可选的音频标签和时间戳）绘制到每'
      + '一帧 RGB 上。0/1 号输入端口与 0 号输出端口共用视频时钟；2 号输入端'
      + '口（音频）自成一个时钟。',
      '把得分最高的 N 個偵測框與標籤（以及可選的音訊標籤和時間戳記）繪製到'
      + '每一格 RGB 上。0/1 號輸入埠與 0 號輸出埠共用視訊時脈；2 號輸入埠（'
      + '音訊）自成一個時脈。'],
  'cfg.detection-overlay.top_n': ['',
      '最多绘制多少个框；0 = 全部',
      '最多繪製多少個框；0 = 全部'],
  'cfg.detection-overlay.score_threshold': ['',
      '丢弃低于该分数的检测结果',
      '丟棄低於該分數的偵測結果'],
  'cfg.detection-overlay.box_thickness': ['',
      '边框像素宽度，限制在 [1,16]',
      '邊框像素寬度，限制在 [1,16]'],
  'cfg.detection-overlay.font_scale': ['',
      '字形整数缩放倍数，限制在 [1,8]',
      '字形整數縮放倍數，限制在 [1,8]'],
  'cfg.detection-overlay.label_padding': ['',
      '标签字形四周的像素留白，限制在 [0,32]',
      '標籤字形四周的像素留白，限制在 [0,32]'],
  'cfg.detection-overlay.score_precision': ['',
      '小数点后位数，限制在 [0,6]',
      '小數點後位數，限制在 [0,6]'],
  'cfg.detection-overlay.input_normalized': ['',
      '仅对 F32 有效；U8 输入时忽略',
      '僅對 F32 有效；U8 輸入時忽略'],
  'cfg.detection-overlay.draw_timestamp': ['',
      '在右下角绘制边带中的时间戳',
      '在右下角繪製邊帶中的時間戳記'],
  'cfg.detection-overlay.timestamp_use_local': ['',
      '时间戳按本地时间而非 UTC 格式化',
      '時間戳記按本地時間而非 UTC 格式化'],
  'cfg.detection-overlay.audio_label_threshold': ['',
      '只有音频标签的分数超过该值时才显示，取值 [0,1]',
      '只有音訊標籤的分數超過該值時才顯示，取值 [0,1]'],
  'cfg.detection-overlay.class_names': ['',
      '以 class_id 为键的标签查找表',
      '以 class_id 為鍵的標籤查找表'],
  'cfg.detection-overlay.oport_capacity': ['',
      '输出环形缓冲容量',
      '輸出環形緩衝容量'],
  'port.detection-overlay.in.frames': ['',
      'RGB TensorBeat [3,H,W]（F32 或 U8）',
      'RGB TensorBeat [3,H,W]（F32 或 U8）'],
  'port.detection-overlay.detections': ['',
      'FlexData 形式的 YOLO/跟踪器检测结果，与帧一一对应',
      'FlexData 形式的 YOLO/追蹤器偵測結果，與影格一一對應'],
  'port.detection-overlay.audio_tags': ['',
      '可选的 FlexData 音频标注流（独立时钟，按时间戳配对）',
      '可選的 FlexData 音訊標註串流（獨立時脈，按時間戳記配對）'],
  'port.detection-overlay.out.frames': ['',
      '已绘制检测框与标签的 TensorBeat（dtype 不变，内存连续）',
      '已繪製偵測框與標籤的 TensorBeat（dtype 不變，記憶體連續）'],

  // ---- Realtime VQA (vision) ----
  'stage.realtime-vqa.name': ['', '实时视觉问答', '即時視覺問答'],
  'stage.realtime-vqa.doc': ['',
      '实时场景视觉问答：将视频帧累积为场景（按间隔/空闲计时边界划分），然'
      + '后对每个场景预填充一次并（批量）解码所配置的问题，输出 FlexData 答'
      + '案包。',
      '即時場景視覺問答：將視訊幀累積為場景（依間隔/閒置計時邊界劃分），然'
      + '後對每個場景預填充一次並（批次）解碼所配置的問題，輸出 FlexData 答'
      + '案包。'],
  'cfg.realtime-vqa.hf_dir': ['',
      'VLM 模型：models 数据库中的键（由 model-fetch 注册）或 HF 风格的模型'
      + '目录；同名时数据库键优先于路径。',
      'VLM 模型：models 資料庫中的鍵（由 model-fetch 註冊）或 HF 風格的模型'
      + '目錄；同名時資料庫鍵優先於路徑。'],
  'cfg.realtime-vqa.coreml_vision_path': ['',
      '预转换的 CoreML 视觉塔（Qwen3.5-VL 或 Gemma-4 的软 token 导出版）：'
      + 'models 数据库中的键（由 model-fetch 注册）或 mlpackage 路径；同名'
      + '时数据库键优先于路径。设置后它会取代 GPU 视觉塔，预处理由该模型自'
      + '身的输入尺寸/格式决定，因此 vlm_input_width/height 与 '
      + 'vlm_max_soft_tokens 会被忽略。',
      '預轉換的 CoreML 視覺塔（Qwen3.5-VL 或 Gemma-4 的軟 token 匯出版）：'
      + 'models 資料庫中的鍵（由 model-fetch 註冊）或 mlpackage 路徑；同名'
      + '時資料庫鍵優先於路徑。設定後它會取代 GPU 視覺塔，前處理由該模型自'
      + '身的輸入尺寸/格式決定，因此 vlm_input_width/height 與 '
      + 'vlm_max_soft_tokens 會被忽略。'],
  'cfg.realtime-vqa.compute_dtype': ['',
      'bf16 | f16 | f32',
      'bf16 | f16 | f32'],
  'cfg.realtime-vqa.language': ['',
      '内置场景提示词的 IETF 界面/提示语言（en-us | zh-cn | zh-tw）；留空则'
      + '继承会话语言。',
      '內建場景提示詞的 IETF 介面/提示語言（en-us | zh-cn | zh-tw）；留空則'
      + '繼承工作階段語言。'],
  'cfg.realtime-vqa.page_tokens': ['',
      'ContextManager 的 K/V 分页大小',
      'ContextManager 的 K/V 分頁大小'],
  'cfg.realtime-vqa.max_new_tokens': ['',
      '每个问题的生成预算（token 数）',
      '每個問題的生成預算（token 數）'],
  'cfg.realtime-vqa.vlm_input_width': ['',
      '固定的 VLM 输入宽度；视觉塔在 GPU 上（非 CoreML）时，会用 GPU 信箱缩'
      + '放到该尺寸（与 vlm_input_height 配对）。0 = 原生尺寸。设置了 '
      + 'coreml_vision_path 时忽略（CoreML 模型会按它自己固定的输入尺寸做信'
      + '箱缩放）。',
      '固定的 VLM 輸入寬度；視覺塔在 GPU 上（非 CoreML）時，會用 GPU 信箱縮'
      + '放到該尺寸（與 vlm_input_height 配對）。0 = 原生尺寸。設定了 '
      + 'coreml_vision_path 時忽略（CoreML 模型會按它自己固定的輸入尺寸做信'
      + '箱縮放）。'],
  'cfg.realtime-vqa.vlm_input_height': ['',
      '固定的 VLM 输入高度；参见 vlm_input_width。0 = 原生尺寸。设置了 '
      + 'coreml_vision_path 时忽略。',
      '固定的 VLM 輸入高度；參見 vlm_input_width。0 = 原生尺寸。設定了 '
      + 'coreml_vision_path 時忽略。'],
  'cfg.realtime-vqa.vlm_max_soft_tokens': ['',
      '每帧视觉软 token 的预算。0 = 沿用编码器静态图/默认的图像预算（细节最'
      + '好）。设为正值可限制每帧的 token 数，用细节换取预填充速度。主要影'
      + '响 Gemma-4（它的稠密视频预算约 64 token/帧，对宽画幅的摄像头画面过'
      + '于粗糙）。例如填 64 即恢复 Gemma 旧的粗糙视频预算。设置了 '
      + 'coreml_vision_path 时忽略（CoreML 模型输出的 token 数是固定的）。',
      '每格視覺軟 token 的預算。0 = 沿用編碼器靜態圖/預設的影像預算（細節最'
      + '好）。設為正值可限制每格的 token 數，用細節換取預填充速度。主要影'
      + '響 Gemma-4（它的稠密影片預算約 64 token/格，對寬畫幅的攝影機畫面過'
      + '於粗糙）。例如填 64 即恢復 Gemma 舊的粗糙影片預算。設定了 '
      + 'coreml_vision_path 時忽略（CoreML 模型輸出的 token 數是固定的）。'],
  'cfg.realtime-vqa.max_frame_gap_ms': ['',
      '结束一个场景的帧间间隔（毫秒）',
      '結束一個場景的幀間間隔（毫秒）'],
  'cfg.realtime-vqa.idle_ticks_to_end': ['',
      '无帧的空闲计时周期数，达到后结束场景（最小为 2）',
      '無幀的閒置計時週期數，達到後結束場景（最小為 2）'],
  'cfg.realtime-vqa.max_frames_per_scene': ['',
      '安全上限；达到后提前结束场景',
      '安全上限；達到後提前結束場景'],
  'cfg.realtime-vqa.catch_up_drop': ['',
      '积压超过该值时每个滴答跳过多少帧；0 = 关闭',
      '積壓超過該值時每個滴答跳過多少影格；0 = 關閉'],
  'cfg.realtime-vqa.batched_decode': ['',
      '批量解码共享前缀的问题分支',
      '批次解碼共享前綴的問題分支'],
  'cfg.realtime-vqa.pipelined_decode': ['',
      'GPU 常驻的流水线化单分支解码（metal）：每个 token 上让主机与 GPU 的'
      + '工作重叠。只影响单分支解码；多问题路径请参见 '
      + 'pipelined_batched_decode。',
      'GPU 常駐的管線化單分支解碼（metal）：每個 token 上讓主機與 GPU 的工'
      + '作重疊。只影響單分支解碼；多問題路徑請參見 '
      + 'pipelined_batched_decode。'],
  'cfg.realtime-vqa.pipelined_batched_decode': ['',
      '需显式开启的 GPU 常驻流水线化批量解码（metal，定长 N 并带前瞻）：'
      + 'token 留在设备上，主机侧的输出与下一步的 CPU 编码与 GPU 重叠。默认'
      + '关闭：答案长度参差不齐时，逐步收缩的同步路径反而更快。',
      '需明確開啟的 GPU 常駐管線化批次解碼（metal，定長 N 並帶前瞻）：token'
      + ' 留在裝置上，主機側的輸出與下一步的 CPU 編碼與 GPU 重疊。預設關閉'
      + '：答案長度參差不齊時，逐步收縮的同步路徑反而更快。'],
  'cfg.realtime-vqa.i8_prefill': ['',
      '加速模式（有损）：预填充 GEMM 改用动态 int8，在矩阵核心 GPU 上速率约'
      + '为 f16 的 2 倍，精度即 int8 水平（开启后预填充不再 token 精确；但'
      + '在这里很合适——realtime-vqa 每个场景都会重新预填充；没有 NAX '
      + 'matmul2d（矩阵核心 GPU + 相应内核）时会被忽略）。默认 false；'
      + 'VPIPE_I8_GEMM 可覆盖。',
      '加速模式（失真）：預填充 GEMM 改用動態 int8，在矩陣核心 GPU 上速率約'
      + '為 f16 的 2 倍，精度即 int8 水準（開啟後預填充不再 token 精確；但'
      + '在這裡很合適——realtime-vqa 每個場景都會重新預填充；沒有 NAX '
      + 'matmul2d（矩陣核心 GPU + 相應核心）時會被忽略）。預設 false；'
      + 'VPIPE_I8_GEMM 可覆寫。'],
  'cfg.realtime-vqa.scene_overlap': ['',
      '把上一场景最后一帧（或最后一对帧）的视觉 token 作为下一场景的第一帧'
      + '重新加入，但仅限时间上连续的场景；设为 false 可禁用',
      '把上一場景最後一格（或最後一對影格）的視覺 token 作為下一場景的第一'
      + '格重新加入，但僅限時間上連續的場景；設為 false 可停用'],
  'cfg.realtime-vqa.video_fps': ['',
      '没有 timestamp_us 时作为后备的标记节奏',
      '沒有 timestamp_us 時作為後備的標記節奏'],
  'cfg.realtime-vqa.disable_thinking': ['',
      '覆盖聊天模板的思考（thinking）默认设置',
      '覆寫聊天範本的思考（thinking）預設設定'],
  'cfg.realtime-vqa.questions': ['',
      '每个场景提出的问题：字符串或字符串数组',
      '每個場景提出的問題：字串或字串陣列'],
  'cfg.realtime-vqa.question_preamble': ['',
      '添加到每个问题回合前的指令（用于引导答案格式）；留空则禁用。',
      '加在每個問題回合前的指令（用於引導答案格式）；留空則停用。'],
  'port.realtime-vqa.frames': ['',
      '平面 u8 RGB TensorBeat [3,H,W]；边带 timestamp_us 驱动场景边界',
      '平面 u8 RGB TensorBeat [3,H,W]；邊帶 timestamp_us 驅動場景邊界'],
  'port.realtime-vqa.trigger': ['',
      '周期性的 TriggerBeat；连续两个空闲滴答会结束一个场景',
      '週期性的 TriggerBeat；連續兩個閒置滴答會結束一個場景'],
  'port.realtime-vqa.audio': ['',
      '可选的 FlexData 音频标签，或 PCM TensorBeat（不限类型）；按时间戳配'
      + '对取用',
      '可選的 FlexData 音訊標籤，或 PCM TensorBeat（不限型別）；按時間戳記'
      + '配對取用'],
  'port.realtime-vqa.sampler': ['',
      '可选的 token 采样器配置 FlexData（sampler-select）。在第一帧时锁存，'
      + '之后每个场景都沿用；未接线 = 贪心（argmax）解码',
      '可選的 token 取樣器組態 FlexData（sampler-select）。在第一格時鎖存，'
      + '之後每個場景都沿用；未接線 = 貪婪（argmax）解碼'],
  'port.realtime-vqa.scene': ['',
      '每个已结束场景的 FlexData：问题 + 答案 + 帧/时间戳元数据',
      '每個已結束場景的 FlexData：問題 + 答案 + 幀/時間戳記中繼資料'],

  // ---- Visual Q&A (vision) ----
  'stage.visual-qa.name': ['', '视觉问答', '視覺問答'],
  'stage.visual-qa.doc': ['',
      '汇聚节点：加载视觉语言模型，编码传入的图像/视频帧，并在每轮提出所配'
      + '置的问题，将答案流式传输到界面。无输出端口。',
      '匯聚節點：載入視覺語言模型，編碼傳入的影像/視訊幀，並在每輪提出所配'
      + '置的問題，將答案串流到介面。無輸出埠。'],
  'cfg.visual-qa.hf_dir': ['',
      'VLM 模型：models 数据库中的键（由 model-fetch 注册）或 HF 风格的模型'
      + '目录；同名时数据库键优先于路径',
      'VLM 模型：models 資料庫中的鍵（由 model-fetch 註冊）或 HF 風格的模型'
      + '目錄；同名時資料庫鍵優先於路徑'],
  'cfg.visual-qa.coreml_vision_path': ['',
      '预转换的 CoreML 视觉塔路径',
      '預轉換的 CoreML 視覺塔路徑'],
  'cfg.visual-qa.coreml_compute_units': ['',
      'CoreML 计算单元（0=CPU，1=+GPU，2=全部，3=+ANE）',
      'CoreML 運算單元（0=CPU，1=+GPU，2=全部，3=+ANE）'],
  'cfg.visual-qa.compute_dtype': ['',
      'bf16 | f16 | f32',
      'bf16 | f16 | f32'],
  'cfg.visual-qa.page_tokens': ['',
      'ContextManager 的 K/V 分页大小',
      'ContextManager 的 K/V 分頁大小'],
  'cfg.visual-qa.max_pages': ['',
      '每个语言模型的分页池容量（>= 1）',
      '每個語言模型的分頁池容量（>= 1）'],
  'cfg.visual-qa.max_new_tokens': ['',
      '每个问题的生成预算（>= 1）',
      '每個問題的生成預算（>= 1）'],
  'cfg.visual-qa.num_images': ['',
      '每轮问答使用多少张图片/帧（>= 1）',
      '每輪問答使用多少張影像/影格（>= 1）'],
  'cfg.visual-qa.pause_ms_between_rounds': ['',
      '每轮问答之后休眠多少毫秒',
      '每輪問答之後休眠多少毫秒'],
  'cfg.visual-qa.batched_decode': ['',
      '批量解码共享前缀的问题分支',
      '批次解碼共用前綴的問題分支'],
  'cfg.visual-qa.i8_prefill': ['',
      '加速模式（有损）：预填充 GEMM 改用动态 int8，在矩阵核心 GPU 上速率约'
      + '为 f16 的 2 倍，精度即 int8 水平（开启后预填充不再 token 精确；没'
      + '有 NAX matmul2d（矩阵核心 GPU + 相应内核）时会被忽略）。默认 false'
      + '；VPIPE_I8_GEMM 可覆盖。',
      '加速模式（失真）：預填充 GEMM 改用動態 int8，在矩陣核心 GPU 上速率約'
      + '為 f16 的 2 倍，精度即 int8 水準（開啟後預填充不再 token 精確；沒'
      + '有 NAX matmul2d（矩陣核心 GPU + 相應核心）時會被忽略）。預設 false'
      + '；VPIPE_I8_GEMM 可覆寫。'],
  'cfg.visual-qa.pre_image_prompt': ['',
      '图像块之前的用户回合文本',
      '影像區塊之前的使用者回合文字'],
  'cfg.visual-qa.post_image_prompt': ['',
      '图像块之后的用户回合文本',
      '影像區塊之後的使用者回合文字'],
  'cfg.visual-qa.decode_after_post_image': ['',
      '在 post-image 之后先解码一次回复，然后再分支',
      '在 post-image 之後先解碼一次回覆，然後再分支'],
  'cfg.visual-qa.disable_thinking': ['',
      '覆盖聊天模板的思考（thinking）默认设置',
      '覆寫聊天範本的思考（thinking）預設設定'],
  'cfg.visual-qa.video': ['',
      '视频模式：{ enabled:bool, fps:real }',
      '視訊模式：{ enabled:bool, fps:real }'],
  'cfg.visual-qa.questions': ['',
      '每轮提出的问题：字符串或字符串数组',
      '每輪提出的問題：字串或字串陣列'],
  'port.visual-qa.images': ['',
      'RGB TensorBeat [3,H,W]；每轮问答 num_images 张（视频模式下则是帧）',
      'RGB TensorBeat [3,H,W]；每輪問答 num_images 張（視訊模式下則是影格）'],
  'port.visual-qa.sampler': ['',
      '可选的 token 采样器配置 FlexData（sampler-select）。在第一轮时锁存，'
      + '之后每一轮都沿用；未接线 = 贪心（argmax）解码',
      '可選的 token 取樣器組態 FlexData（sampler-select）。在第一輪時鎖存，'
      + '之後每一輪都沿用；未接線 = 貪婪（argmax）解碼'],

  // ---- YOLO Detector (vision) ----
  'stage.yolo-detection.name': ['', '目标检测器', '物件偵測器'],
  'stage.yolo-detection.doc': ['',
      '对每一帧 RGB 运行 YOLOv6 结构的 CoreML 检测器（信箱缩放 + NMS），每'
      + '帧发出一条 FlexData 检测记录。',
      '對每一格 RGB 執行 YOLOv6 結構的 CoreML 偵測器（信箱縮放 + NMS），每'
      + '格發出一筆 FlexData 偵測記錄。'],
  'cfg.yolo-detection.model_path': ['',
      'YOLO CoreML 模型：models 数据库中的键（由 model-fetch 注册）或 '
      + 'mlpackage 路径；同名时数据库键优先于路径',
      'YOLO CoreML 模型：models 資料庫中的鍵（由 model-fetch 註冊）或 '
      + 'mlpackage 路徑；同名時資料庫鍵優先於路徑'],
  'cfg.yolo-detection.input_feature_name': ['',
      '模型输入特征名；留空（默认）自动匹配模型唯一的输入',
      '模型輸入特徵名；留空（預設）自動比對模型唯一的輸入'],
  'cfg.yolo-detection.output_feature_name': ['',
      '模型输出特征名；留空（默认）自动匹配模型唯一的输出',
      '模型輸出特徵名；留空（預設）自動比對模型唯一的輸出'],
  'cfg.yolo-detection.num_classes': ['',
      '类别数量；缺省或为 0 时自动检测',
      '類別數量；缺少或為 0 時自動偵測'],
  'cfg.yolo-detection.confidence_threshold': ['',
      '丢弃低于该分数的检测结果',
      '丟棄低於該分數的偵測結果'],
  'cfg.yolo-detection.iou_threshold': ['',
      '逐类别 NMS 的 IoU 阈值',
      '逐類別 NMS 的 IoU 閾值'],
  'cfg.yolo-detection.max_detections': ['',
      'NMS 之后最多保留多少个检测',
      'NMS 之後最多保留多少個偵測'],
  'cfg.yolo-detection.class_names': ['',
      '标签名称；长度必须与 num_classes 一致',
      '標籤名稱；長度必須與 num_classes 一致'],
  'cfg.yolo-detection.compute_units': ['',
      '0=仅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE',
      '0=僅 CPU  1=CPU+GPU  2=全部  3=CPU+ANE'],
  'cfg.yolo-detection.uses_cpu_only': ['',
      '推理时对 compute_units 的覆盖',
      '推理時對 compute_units 的覆寫'],
  'cfg.yolo-detection.oport_capacity': ['',
      '输出环形缓冲容量',
      '輸出環形緩衝容量'],
  'port.yolo-detection.frames': ['',
      'RGB TensorBeat [3,H,W]，f32，取值 [0,1]',
      'RGB TensorBeat [3,H,W]，f32，取值 [0,1]'],
  'port.yolo-detection.detections': ['',
      'FlexData '
      + '{frame_width,frame_height,detections[{class_id,class_name,score,x1,y1,x2,y2}]}',
      'FlexData '
      + '{frame_width,frame_height,detections[{class_id,class_name,score,x1,y1,x2,y2}]}'],

  // ---- Compare Images (visual) ----
  'stage.compare-image.name': ['', '图像对比', '影像對比'],
  'stage.compare-image.doc': ['',
      '汇聚节点：把两张图片配对，在它自己的 web 界面视图中比较——只看 A、只'
      + '看 B、并排（左右或上下、中间带分隔线），或者可拖动的垂直/水平擦除'
      + '对比。缩放和平移在两张图之间保持同步。分辨率不一致时会统一到 max('
      + 'Wa,Wb) x max(Ha,Hb)，按填充策略处理（缩放适配、居中、pad_color 边'
      + '框）。缺失的输入显示为黑色。无输出端口。',
      '匯聚節點：把兩張圖片配對，在它自己的 web 介面檢視中比較——只看 A、只'
      + '看 B、並排（左右或上下、中間帶分隔線），或者可拖曳的垂直/水平擦除'
      + '對比。縮放和平移在兩張圖之間保持同步。解析度不一致時會統一到 max('
      + 'Wa,Wb) x max(Ha,Hb)，按填充策略處理（縮放調整、置中、pad_color 邊'
      + '框）。缺少的輸入顯示為黑色。無輸出埠。'],
  'cfg.compare-image.input_normalized': ['',
      'F32 输入是 [0,1] 而非 [0,255]',
      'F32 輸入是 [0,1] 而非 [0,255]'],
  'cfg.compare-image.title': ['',
      '在 web 界面对比选择器中显示的可选标签；留空 = 使用阶段 id',
      '在 web 介面對比選擇器中顯示的可選標籤；留空 = 使用階段 id'],
  'cfg.compare-image.pad_color': ['',
      '#RRGGBB，用于填充较小那张图在统一尺寸后留下的边框',
      '#RRGGBB，用於填充較小那張圖在統一尺寸後留下的邊框'],
  'port.compare-image.a': ['',
      '图像 A：平面 RGB TensorBeat [3,H,W]（F32 或 U8）。可选——未接线或没有'
      + '数据时显示为黑色。',
      '影像 A：平面 RGB TensorBeat [3,H,W]（F32 或 U8）。可選——未接線或沒有'
      + '資料時顯示為黑色。'],
  'port.compare-image.b': ['',
      '图像 B：平面 RGB TensorBeat [3,H,W]（F32 或 U8）。可选——未接线或没有'
      + '数据时显示为黑色。',
      '影像 B：平面 RGB TensorBeat [3,H,W]（F32 或 U8）。可選——未接線或沒有'
      + '資料時顯示為黑色。'],

  // ---- Create Mask (visual) ----
  'stage.create-mask.name': ['', '创建遮罩', '建立遮罩'],
  'stage.create-mask.doc': ['',
      '制作遮罩——既可以在它自己的 web 界面编辑器里用圆形画笔手绘（半径可调'
      + '，alpha 模式下还可调硬度），也可以直接从输入透传。支持两态、软 '
      + 'alpha 和多类别三种模式。每次提交发出一个节拍：要么是遮罩 [1,H,W]，'
      + '要么是把遮罩绘制其上的参考图像 [3,H,W]。把 `interactive` 设为 '
      + 'false 则无界面运行，收到遮罩即叠加输出。2 个可选输入端口，1 个输出'
      + '端口。',
      '製作遮罩——既可以在它自己的 web 介面編輯器裡用圓形筆刷手繪（半徑可調'
      + '，alpha 模式下還可調硬度），也可以直接從輸入透傳。支援兩態、軟 '
      + 'alpha 和多類別三種模式。每次提交發出一個節拍：要麼是遮罩 [1,H,W]，'
      + '要麼是把遮罩繪製其上的參考影像 [3,H,W]。把 `interactive` 設為 '
      + 'false 則無介面執行，收到遮罩即疊加輸出。2 個可選輸入埠，1 個輸出埠'
      + '。'],
  'cfg.create-mask.mask_mode': ['',
      '"binary"（两态 0/255）、"alpha"（0..255 的软覆盖度，画笔硬度决定过渡'
      + '形状），或 "class"（多类别索引图）',
      '"binary"（兩態 0/255）、"alpha"（0..255 的軟覆蓋度，筆刷硬度決定過渡'
      + '形狀），或 "class"（多類別索引圖）'],
  'cfg.create-mask.classes': ['',
      '类别数量，含背景（索引 0）；仅 "class" 模式使用',
      '類別數量，含背景（索引 0）；僅 "class" 模式使用'],
  'cfg.create-mask.class_colors': ['',
      '以逗号分隔的 #RRGGBB，每个类别一个。仅用于呈现——它决定编辑器和叠加显'
      + '示的样子，绝不影响离开本阶段的索引图。留空 = 使用内置调色板',
      '以逗號分隔的 #RRGGBB，每個類別一個。僅用於呈現——它決定編輯器和疊加顯'
      + '示的樣子，絕不影響離開本階段的索引圖。留空 = 使用內建調色盤'],
  'cfg.create-mask.output': ['',
      '"mask" 发出遮罩 [1,H,W]；"overlay" 发出把遮罩绘制其上的参考图像 [3,H'
      + ',W]',
      '"mask" 發出遮罩 [1,H,W]；"overlay" 發出把遮罩繪製其上的參考影像 [3,H'
      + ',W]'],
  'cfg.create-mask.overlay_color': ['',
      'binary / alpha 模式下绘制遮罩所用的 #RRGGBB；class 模式使用 '
      + 'class_colors',
      'binary / alpha 模式下繪製遮罩所用的 #RRGGBB；class 模式使用 '
      + 'class_colors'],
  'cfg.create-mask.overlay_opacity': ['',
      '遮罩绘制到参考图像上的强度，0..1',
      '遮罩繪製到參考影像上的強度，0..1'],
  'cfg.create-mask.output_dtype': ['',
      '"u8"（0..255，或一个类别索引）或 "f32"（覆盖度，取值 [0,1]；类别索引'
      + '仍是索引）',
      '"u8"（0..255，或一個類別索引）或 "f32"（覆蓋度，取值 [0,1]；類別索引'
      + '仍是索引）'],
  'cfg.create-mask.width': ['',
      '遮罩画布宽度；0 表示由 `height` 和参考图像的宽高比推断，或直接与参考'
      + '图像一致',
      '遮罩畫布寬度；0 表示由 `height` 和參考影像的長寬比推斷，或直接與參考'
      + '影像一致'],
  'cfg.create-mask.height': ['',
      '遮罩画布高度；0 的推断方式与 `width` 相同',
      '遮罩畫布高度；0 的推斷方式與 `width` 相同'],
  'cfg.create-mask.interactive': ['',
      '运行遮罩编辑器，每次提交发出一个节拍。设为 false 则没有图形界面：遮'
      + '罩来自 in-mask，收到即发出',
      '執行遮罩編輯器，每次提交發出一個節拍。設為 false 則沒有圖形介面：遮'
      + '罩來自 in-mask，收到即發出'],
  'cfg.create-mask.input_normalized': ['',
      'F32 输入是 [0,1] 而非 [0,255]',
      'F32 輸入是 [0,1] 而非 [0,255]'],
  'cfg.create-mask.title': ['',
      '在 web 界面遮罩编辑器选择器中显示的可选标签；留空 = 使用阶段 id',
      '在 web 介面遮罩編輯器選擇器中顯示的可選標籤；留空 = 使用階段 id'],
  'port.create-mask.ref-image': ['',
      '参考图像：平面 RGB TensorBeat [3,H,W]（F32 或 U8）。可选——它是编辑器'
      + '的背景，也是叠加输出的底图；未接线时编辑器在黑色底上绘制。',
      '參考影像：平面 RGB TensorBeat [3,H,W]（F32 或 U8）。可選——它是編輯器'
      + '的背景，也是疊加輸出的底圖；未接線時編輯器在黑色底上繪製。'],
  'port.create-mask.in-mask': ['',
      '作为起点的遮罩：平面 TensorBeat [1,H,W] 或 [H,W]（F32 或 U8）。可选—'
      + '—它为编辑器提供初值；`interactive` 为 false 时它就是全部输入。',
      '作為起點的遮罩：平面 TensorBeat [1,H,W] 或 [H,W]（F32 或 U8）。可選—'
      + '—它為編輯器提供初值；`interactive` 為 false 時它就是全部輸入。'],
  'port.create-mask.out': ['',
      '按画布分辨率给出的遮罩 [1,H,W]，或按参考图像分辨率给出的、已绘制遮罩'
      + '的参考图像 [3,H,W]——取决于 `output` 的选择。',
      '按畫布解析度給出的遮罩 [1,H,W]，或按參考影像解析度給出的、已繪製遮罩'
      + '的參考影像 [3,H,W]——取決於 `output` 的選擇。'],

  // ---- Resample (visual) ----
  'stage.image-resample.name': ['', '图像重采样', '影像重新取樣'],
  'stage.image-resample.doc': ['',
      '把 rgb-frames 重采样到固定的宽 x 高，宽高比的处理方式（填充 / 裁剪 /'
      + ' 拉伸 / 手动）与填充色均可配置。默认使用 Lanczos-3 重采样；也可选'
      + '更省的 \'bilinear\'。u8 帧走 GPU 信箱缩放内核，f32 帧走同时实现了两'
      + '种滤波器的 CPU 路径。输入端口与输出端口同属一个时钟域（1:1）。',
      '把 rgb-frames 重新取樣到固定的寬 x 高，長寬比的處理方式（填充 / 裁剪'
      + ' / 拉伸 / 手動）與填充色均可設定。預設使用 Lanczos-3 重新取樣；也'
      + '可選更省的 \'bilinear\'。u8 影格走 GPU 信箱縮放核心，f32 影格走同時'
      + '實作了兩種濾波器的 CPU 路徑。輸入埠與輸出埠同屬一個時脈域（1:1）。'],
  'cfg.image-resample.width': ['',
      '输出宽度（像素）；留空（或设为小于等于 0）则由 height 推断，保持源的'
      + '宽高比',
      '輸出寬度（像素）；留空（或設為小於等於 0）則由 height 推斷，保持來源'
      + '的長寬比'],
  'cfg.image-resample.height': ['',
      '输出高度（像素）；留空（或设为小于等于 0）则由 width 推断，保持源的'
      + '宽高比',
      '輸出高度（像素）；留空（或設為小於等於 0）則由 width 推斷，保持來源'
      + '的長寬比'],
  'cfg.image-resample.fit': ['',
      '宽高比处理方式：pad（对齐长边并用 pad_color 填充）| crop（对齐短边并'
      + '居中裁剪）| stretch（改变宽高比）| manual（从 src_x,src_y 起按 '
      + 'scale 采样，其余填充）',
      '長寬比處理方式：pad（對齊長邊並用 pad_color 填充）| crop（對齊短邊並'
      + '置中裁剪）| stretch（改變長寬比）| manual（從 src_x,src_y 起按 '
      + 'scale 取樣，其餘填充）'],
  'cfg.image-resample.pad_color': ['',
      'pad / manual 所用的 #RRGGBB 纯色填充（f32 帧按 0..1 归一化解读）',
      'pad / manual 所用的 #RRGGBB 純色填充（f32 影格按 0..1 正規化解讀）'],
  'cfg.image-resample.src_x': ['',
      'manual：源起点 x',
      'manual：來源起點 x'],
  'cfg.image-resample.src_y': ['',
      'manual：源起点 y',
      'manual：來源起點 y'],
  'cfg.image-resample.scale': ['',
      'manual：重采样比例（每个源像素对应多少输出像素，须大于 0）',
      'manual：重新取樣比例（每個來源像素對應多少輸出像素，須大於 0）'],
  'cfg.image-resample.algorithm': ['',
      '插值算法：\'lanczos\'（默认——Lanczos-3，带抗锯齿，与 PIL 的 LANCZOS 一'
      + '致）| \'bilinear\'（更省，但缩小时会产生混叠）。经过本阶段的绝大多数'
      + '流量都是缩小到某个模型的输入画布，而双线性恰恰保留了本应被滤掉的高'
      + '频——所以好滤波器是默认，快滤波器需要显式选择',
      '插值演算法：\'lanczos\'（預設——Lanczos-3，帶反鋸齒，與 PIL 的 LANCZOS '
      + '一致）| \'bilinear\'（更省，但縮小時會產生疊頻）。經過本階段的絕大多'
      + '數流量都是縮小到某個模型的輸入畫布，而雙線性恰恰保留了本應被濾掉的'
      + '高頻——所以好濾波器是預設，快濾波器需要明確選擇'],
  'port.image-resample.in.frames': ['',
      '平面 RGB TensorBeat [3,H,W]（u8 或 f32）',
      '平面 RGB TensorBeat [3,H,W]（u8 或 f32）'],
  'port.image-resample.out.frames': ['',
      '重采样后的平面 RGB TensorBeat [3,height,width]（dtype 不变）',
      '重新取樣後的平面 RGB TensorBeat [3,height,width]（dtype 不變）'],

  // ---- Load Image (visual) ----
  'stage.load-image.name': ['', '加载图片', '載入影像'],
  'stage.load-image.doc': ['',
      '源：用 FFmpeg 把文件/URL 中的静态图片解码为平面 U8 RGB TensorBeat。'
      + '若接了 trigger 输入端口，则每个节拍发出一张图片；未接时一次发完全'
      + '部然后结束。',
      '來源：用 FFmpeg 把檔案/URL 中的靜態影像解碼為平面 U8 RGB TensorBeat'
      + '。若接了 trigger 輸入埠，則每個節拍發出一張影像；未接時一次發完全'
      + '部然後結束。'],
  'cfg.load-image.url': ['',
      '图片路径/URL 字符串，或字符串数组',
      '影像路徑/URL 字串，或字串陣列'],
  'port.load-image.trigger': ['',
      '可选的节奏节拍（例如 chrono）；每来一个就解码并发出下一张图片',
      '可選的節奏節拍（例如 chrono）；每來一個就解碼並發出下一張影像'],
  'port.load-image.image': ['',
      '解码后的图片，平面 U8 RGB TensorBeat [3,H,W]',
      '解碼後的影像，平面 U8 RGB TensorBeat [3,H,W]'],
  'port.load-image.metadata': ['',
      '随图片一同携带的 FlexData {url, width, height, exif{...}, '
      + 'exif_tiff_b64}；每张图片一个节拍，与 0 号输出端口配对。可接到 '
      + 'save-image 的 metadata 输入端口',
      '隨影像一同攜帶的 FlexData {url, width, height, exif{...}, '
      + 'exif_tiff_b64}；每張影像一個節拍，與 0 號輸出埠配對。可接到 '
      + 'save-image 的 metadata 輸入埠'],

  // ---- Load Video (visual) ----
  'stage.load-video.name': ['', '加载视频', '載入影片'],
  'stage.load-video.doc': ['',
      '源：解复用视频文件或网络 URL，在各自独立的每流时钟上发出编码的视频/'
      + '音频数据包——相当于 rtsp-capture 的文件版。下游接 video-to-rgb / '
      + 'audio-to-pcm，由它们负责解码。',
      '來源：解多工影片檔案或網路 URL，在各自獨立的每串流時脈上發出編碼的視'
      + '訊/音訊封包——相當於 rtsp-capture 的檔案版。下游接 video-to-rgb / '
      + 'audio-to-pcm，由它們負責解碼。'],
  'cfg.load-video.input_url': ['',
      '文件路径或网络 URL（rtsp/http/…）',
      '檔案路徑或網路 URL（rtsp/http/…）'],
  'cfg.load-video.format': ['',
      '强制指定解复用器；"" = 自动检测',
      '強制指定解多工器；"" = 自動偵測'],
  'cfg.load-video.enable_video': ['', '发出视频输出端口', '發出視訊輸出埠'],
  'cfg.load-video.enable_audio': ['', '发出音频输出端口', '發出音訊輸出埠'],
  'cfg.load-video.video_stream_index': ['',
      '要使用的流；-1 = 第一条视频流',
      '要使用的串流；-1 = 第一條視訊串流'],
  'cfg.load-video.audio_stream_index': ['',
      '要使用的流；-1 = 第一条音频流',
      '要使用的串流；-1 = 第一條音訊串流'],
  'cfg.load-video.start_s': ['',
      '读取第一个数据包前先定位到该媒体时间（自文件开头起的秒数）；0（默认'
      + '）从头开始。精度到关键帧：关键帧与 start_s 之间的数据包仍会发出，'
      + '因为其后的帧要靠它们解码',
      '讀取第一個封包前先定位到該媒體時間（自檔案開頭起的秒數）；0（預設）'
      + '從頭開始。精度到關鍵影格：關鍵影格與 start_s 之間的封包仍會發出，'
      + '因為其後的影格要靠它們解碼'],
  'cfg.load-video.duration_s': ['',
      '在 start_s 之后这么多秒处停止；0（默认）读到文件结尾。每条启用的流各'
      + '自越过终点后停止，两条都停止时本次运行结束',
      '在 start_s 之後這麼多秒處停止；0（預設）讀到檔案結尾。每條啟用的串流'
      + '各自越過終點後停止，兩條都停止時本次執行結束'],
  'cfg.load-video.read_timeout_ms': ['',
      '网络读取超时（毫秒）；0 = 不限时',
      '網路讀取逾時（毫秒）；0 = 不限時'],
  'cfg.load-video.options': ['',
      '输入打开时使用的字符串选项 av_dict',
      '輸入開啟時使用的字串選項 av_dict'],
  'port.load-video.video': ['',
      '每个视频数据包一个 EncodedSegment，携带该流的 codec_id / width / '
      + 'height / fps_num / fps_den / extradata。请接到 video-to-rgb，由它'
      + '负责解码',
      '每個視訊封包一個 EncodedSegment，攜帶該串流的 codec_id / width / '
      + 'height / fps_num / fps_den / extradata。請接到 video-to-rgb，由它'
      + '負責解碼'],
  'port.load-video.audio': ['',
      '每个音频数据包一个 EncodedSegment，携带该流的 codec_id / sample_rate'
      + ' / channels / extradata。请接到 audio-to-pcm，由它负责解码与重采样',
      '每個音訊封包一個 EncodedSegment，攜帶該串流的 codec_id / sample_rate'
      + ' / channels / extradata。請接到 audio-to-pcm，由它負責解碼與重新取'
      + '樣'],

  // ---- RGB to Video (visual) ----
  'stage.rgb-to-video.name': ['', 'RGB → 视频', 'RGB → 視訊'],
  'stage.rgb-to-video.doc': ['',
      '把平面 U8 RGB 图像节拍适配成 save-video 编码器所读取的 '
      + 'VideoStreamParams + FrameRef 流。生成式图像格式与 ffmpeg 之间的接'
      + '缝。',
      '把平面 U8 RGB 影像節拍轉接成 save-video 編碼器所讀取的 '
      + 'VideoStreamParams + FrameRef 串流。生成式影像格式與 ffmpeg 之間的'
      + '接縫。'],
  'cfg.rgb-to-video.fps': ['',
      '当生产者的边带未携带帧率时，声明使用的帧率。16 是 Wan 视频的默认值',
      '當生產者的邊帶未攜帶影格率時，宣告使用的影格率。16 是 Wan 影片的預設'
      + '值'],
  'cfg.rgb-to-video.pix_fmt': ['',
      '发出帧的像素格式："yuv420p"（H.264 所需）或 "rgb24"（不做转换，供无'
      + '损编码器使用）',
      '發出影格的像素格式："yuv420p"（H.264 所需）或 "rgb24"（不做轉換，供'
      + '無失真編碼器使用）'],
  'port.rgb-to-video.image': ['',
      '平面 U8 RGB TensorBeat [3, H, W]，按呈现顺序每帧一个',
      '平面 U8 RGB TensorBeat [3, H, W]，按呈現順序每格一個'],
  'port.rgb-to-video.video': ['',
      '先一个 VideoStreamParams 头，然后每帧一个 FrameRef——即 save-video 读'
      + '取的约定',
      '先一個 VideoStreamParams 標頭，然後每格一個 FrameRef——即 save-video '
      + '讀取的約定'],

  // ---- Save Image (visual) ----
  'stage.save-image.name': ['', '保存图片', '儲存影像'],
  'stage.save-image.doc': ['',
      '汇聚节点：用 FFmpeg 把每个平面 U8 RGB TensorBeat [3,H,W] 编码为图片'
      + '文件（PNG/JPEG/WebP/BMP/TIFF）。load-image 的逆过程；质量/压缩旋钮'
      + '控制落盘方式。',
      '匯聚節點：用 FFmpeg 把每個平面 U8 RGB TensorBeat [3,H,W] 編碼為影像'
      + '檔（PNG/JPEG/WebP/BMP/TIFF）。load-image 的逆過程；品質/壓縮旋鈕控'
      + '制存檔方式。'],
  'cfg.save-image.path': ['',
      '输出图片文件路径；路径中带 printf 整数格式（如 frame-%04d.png）时用'
      + '于给图片序列编号，否则后续图片会自动加 -NNNNNN 后缀',
      '輸出影像檔路徑；路徑中帶 printf 整數格式（如 frame-%04d.png）時用於'
      + '給影像序列編號，否則後續影像會自動加 -NNNNNN 後綴'],
  'cfg.save-image.format': ['',
      'png | jpeg (jpg) | webp | bmp | tiff；默认取自路径扩展名，否则为 png',
      'png | jpeg (jpg) | webp | bmp | tiff；預設取自路徑副檔名，否則為 png'],
  'cfg.save-image.quality': ['',
      '有损编码（jpeg、有损 webp）：1..100，越大越好（默认 90）',
      '失真編碼（jpeg、失真 webp）：1..100，越大越好（預設 90）'],
  'cfg.save-image.compression': ['',
      'PNG 的 zlib 压缩级别 0..9，越大文件越小、越慢（默认 6）',
      'PNG 的 zlib 壓縮等級 0..9，越大檔案越小、越慢（預設 6）'],
  'cfg.save-image.lossless': ['',
      'webp 无损模式（默认 false）',
      'webp 無失真模式（預設 false）'],
  'port.save-image.image': ['',
      '平面 U8 RGB TensorBeat [3,H,W]（load-image / vae-decode 的格式）',
      '平面 U8 RGB TensorBeat [3,H,W]（load-image / vae-decode 的格式）'],
  'port.save-image.metadata': ['',
      '可选的 FlexData，携带 exif_tiff_b64（即 load-image 的 metadata 输出'
      + '端口所发的内容）；接上后会把 EXIF 块写入每个输出文件。每张图片一个'
      + '节拍',
      '可選的 FlexData，攜帶 exif_tiff_b64（即 load-image 的 metadata 輸出'
      + '埠所發的內容）；接上後會把 EXIF 區塊寫入每個輸出檔案。每張影像一個'
      + '節拍'],

  // ---- Save Video (visual) ----
  'stage.save-video.name': ['', '保存视频', '儲存影片'],
  'stage.save-video.doc': ['',
      '汇聚节点：把传入的视频/音频帧编码（默认 H.264 + AAC）并封装进容器文'
      + '件。无输出端口；各输入端口运行在各自独立的每流时钟上。',
      '匯聚節點：把傳入的視訊/音訊影格編碼（預設 H.264 + AAC）並封裝進容器'
      + '檔案。無輸出埠；各輸入埠執行在各自獨立的每串流時脈上。'],
  'cfg.save-video.output_url': ['',
      '输出文件路径 / URL',
      '輸出檔案路徑 / URL'],
  'cfg.save-video.format': ['',
      '容器格式；"" = 自动推断',
      '容器格式；"" = 自動推斷'],
  'cfg.save-video.enable_video': ['', '编码一条视频流', '編碼一條視訊串流'],
  'cfg.save-video.enable_audio': ['', '编码一条音频流', '編碼一條音訊串流'],
  'cfg.save-video.video_codec': ['',
      '视频编码器。默认值是 macOS 自带的、硬件加速的编码器，且不带授权义务—'
      + '—libx264 属 GPL，无法随本项目再分发的 FFmpeg 一起发布。非 Apple 构'
      + '建会在初始化时回退到 libx264。所链接的 libavcodec 提供的任何编码器'
      + '都可接受',
      '視訊編碼器。預設值是 macOS 內建、硬體加速的編碼器，且不帶授權義務——'
      + 'libx264 屬 GPL，無法隨本專案再散布的 FFmpeg 一起發布。非 Apple 建'
      + '置會在初始化時回退到 libx264。所連結的 libavcodec 提供的任何編碼器'
      + '都可接受'],
  'cfg.save-video.video_bitrate': ['',
      '目标视频码率，单位比特/秒。以恒定质量模式运行的编码器会忽略它——参见 '
      + 'video_crf',
      '目標視訊位元率，單位位元/秒。以恆定品質模式執行的編碼器會忽略它——參'
      + '見 video_crf'],
  'cfg.save-video.video_preset': ['',
      '编码速度/体积预设（ultrafast .. veryslow），仅对接受该选项的编码器有'
      + '效；其余忽略',
      '編碼速度/體積預設（ultrafast .. veryslow），僅對接受該選項的編碼器有'
      + '效；其餘忽略'],
  'cfg.save-video.video_crf': ['',
      '以恒定质量目标代替码率：越小越好，0 为无损，常用范围约 18-28。-1（默'
      + '认）表示不设置，由 video_bitrate 决定。仅支持 crf 的编码器会采用，'
      + '其余忽略',
      '以恆定品質目標取代位元率：越小越好，0 為無失真，常用範圍約 18-28。-1'
      + '（預設）表示不設定，由 video_bitrate 決定。僅支援 crf 的編碼器會採'
      + '用，其餘忽略'],
  'cfg.save-video.video_gop_size': ['',
      '关键帧间隔（帧数）。越小则跳转与恢复越快，代价是体积变大',
      '關鍵影格間隔（影格數）。越小則跳轉與復原越快，代價是體積變大'],
  'cfg.save-video.video_options': ['',
      '传给视频编码器的额外 av_dict 选项，用于上面各项没有涵盖的设置',
      '傳給視訊編碼器的額外 av_dict 選項，用於上面各項沒有涵蓋的設定'],
  'cfg.save-video.audio_codec': ['', '音频编码器', '音訊編碼器'],
  'cfg.save-video.audio_bitrate': ['',
      '目标音频码率，单位比特/秒',
      '目標音訊位元率，單位位元/秒'],
  'cfg.save-video.audio_options': ['',
      '传给音频编码器的额外 av_dict 选项',
      '傳給音訊編碼器的額外 av_dict 選項'],
  'cfg.save-video.muxer_options': ['',
      '传给封装器的字符串选项 av_dict。没有展平成独立键：与上面的编码器设置'
      + '不同，这些选项没有固定的名字集合——取决于所选的容器',
      '傳給封裝器的字串選項 av_dict。沒有攤平成獨立鍵：與上面的編碼器設定不'
      + '同，這些選項沒有固定的名稱集合——取決於所選的容器'],
  'cfg.save-video.video': ['',
      '已被 video_codec / video_bitrate / video_preset / video_crf / '
      + 'video_gop_size / video_options 取代。仍会读取，但会给出警告；同时'
      + '设置的扁平键优先',
      '已被 video_codec / video_bitrate / video_preset / video_crf / '
      + 'video_gop_size / video_options 取代。仍會讀取，但會給出警告；同時'
      + '設定的扁平鍵優先'],
  'cfg.save-video.audio': ['',
      '已被 audio_codec / audio_bitrate / audio_options 取代。仍会读取，但'
      + '会给出警告；同时设置的扁平键优先',
      '已被 audio_codec / audio_bitrate / audio_options 取代。仍會讀取，但'
      + '會給出警告；同時設定的扁平鍵優先'],
  'port.save-video.video': ['',
      '先一个 VideoStreamParams 头，随后是视频 FrameRef',
      '先一個 VideoStreamParams 標頭，隨後是視訊 FrameRef'],
  'port.save-video.audio': ['',
      '先一个 AudioStreamParams 头，随后是音频 FrameRef；或者带 sample_rate'
      + ' 边带的 f32 PCM TensorBeat [声道数, 采样数]',
      '先一個 AudioStreamParams 標頭，隨後是音訊 FrameRef；或者帶 '
      + 'sample_rate 邊帶的 f32 PCM TensorBeat [聲道數, 取樣數]'],

  // ---- Frame Dropper (visual) ----
  'stage.temporal-decimation.name': ['', '视频抽帧', '影片抽幀'],
  'stage.temporal-decimation.doc': ['',
      '令牌桶式抽帧器：在平均 FPS 上限之内保留帧，并偏向有运动的帧（1 号输'
      + '入端口接了检测结果时，则偏向重点类别的运动）。保留下来的帧原样转发'
      + '。',
      '權杖桶式抽格器：在平均 FPS 上限之內保留影格，並偏向有動態的影格（1 '
      + '號輸入埠接了偵測結果時，則偏向重點類別的動態）。保留下來的影格原樣'
      + '轉發。'],
  'cfg.temporal-decimation.max_avg_fps': ['',
      '保留速率的上限；允许小于 1',
      '保留速率的上限；允許小於 1'],
  'cfg.temporal-decimation.target_fps': ['',
      'max_avg_fps 的旧别名',
      'max_avg_fps 的舊別名'],
  'cfg.temporal-decimation.max_consecutive_drops': ['',
      '强制保留的下限：最多可以连续丢弃多少帧',
      '強制保留的下限：最多可以連續丟棄多少影格'],
  'cfg.temporal-decimation.tile_w': ['',
      '运动特征分块的列数，限制在 [1,1024]',
      '動態特徵分塊的行數，限制在 [1,1024]'],
  'cfg.temporal-decimation.tile_h': ['',
      '运动特征分块的行数，限制在 [1,1024]',
      '動態特徵分塊的列數，限制在 [1,1024]'],
  'cfg.temporal-decimation.motion_threshold': ['',
      '优先级达到该门槛以上的帧才可能被保留',
      '優先度達到該門檻以上的影格才可能被保留'],
  'cfg.temporal-decimation.focus_motion_gain': ['',
      '重点类别运动占比的权重',
      '重點類別動態占比的權重'],
  'cfg.temporal-decimation.bucket_capacity': ['',
      '令牌桶的突发上限，须大于等于 1',
      '權杖桶的突發上限，須大於等於 1'],
  'cfg.temporal-decimation.focus_classes': ['',
      '重点类别列表，可混用整数 class_id 与字符串 class_name',
      '重點類別清單，可混用整數 class_id 與字串 class_name'],
  'port.temporal-decimation.in.frames': ['',
      '平面 u8 RGB TensorBeat [3,H,W]',
      '平面 u8 RGB TensorBeat [3,H,W]'],
  'port.temporal-decimation.detections': ['',
      '可选的 FlexData YOLO 检测结果，与帧一一对应读取',
      '可選的 FlexData YOLO 偵測結果，與影格一一對應讀取'],
  'port.temporal-decimation.out.frames': ['',
      '被保留的帧（原样转发的 TensorBeat）；被丢弃的帧不产生任何输出',
      '被保留的影格（原樣轉發的 TensorBeat）；被丟棄的影格不產生任何輸出'],

  // ---- Temporal Resample (visual) ----
  'stage.temporal-resample.name': ['', '时间重采样', '時間重新取樣'],
  'stage.temporal-resample.doc': ['',
      '通过一条 FFmpeg 滤镜链把平面 RGB 帧从一个帧率重采样到另一个帧率，并'
      + '可选择被丢弃的信息如何处理：丢帧/复帧、混合、盒式平均（运动模糊）'
      + '或运动补偿插值。它是 image-resample 在时间轴上的对应物。',
      '透過一條 FFmpeg 濾鏡鏈把平面 RGB 影格從一個影格率重新取樣到另一個影'
      + '格率，並可選擇被丟棄的資訊如何處理：丟格/複格、混合、盒式平均（動'
      + '態模糊）或運動補償插值。它是 image-resample 在時間軸上的對應物。'],
  'cfg.temporal-resample.output_fps': ['',
      '要重采样到的目标帧率。必填且为正——它就是这条请求的全部，一个给它设默'
      + '认值的阶段无非是顶着重采样器名字的直通。它会用连分数转换成真正的有'
      + '理数，因此 29.97 到达滤镜时是 30000/1001 而不是 2997/100：滤镜正是'
      + '拿它做除法的，而一个差之毫厘的帧率每几千帧就会丢一帧，还什么都不说',
      '要重新取樣到的目標影格率。必填且為正——它就是這條請求的全部，一個給它'
      + '設預設值的階段無非是頂著重新取樣器名字的直通。它會用連分數轉換成真'
      + '正的有理數，因此 29.97 抵達濾鏡時是 30000/1001 而不是 2997/100：濾'
      + '鏡正是拿它做除法的，而一個差之毫釐的影格率每幾千格就會丟一格，還什'
      + '麼都不說'],
  'cfg.temporal-resample.input_fps': ['',
      '源帧率，用于节拍本身没有说明的情形。片段节拍带 `fps`，逐帧节拍带 `'
      + 'fps_num`/`fps_den`，但只有在源知道的时候才有；这一项是后备值，而且'
      + '一旦设置就会覆盖边带。设错会改变比例，于是片段的时长不对，而其中每'
      + '一帧看上去都没问题。0（默认）：采用节拍自己的值',
      '源影格率，用於節拍本身沒有說明的情形。片段節拍帶 `fps`，逐格節拍帶 `'
      + 'fps_num`/`fps_den`，但只有在來源知道的時候才有；這一項是後備值，而'
      + '且一旦設定就會覆寫邊帶。設錯會改變比例，於是片段的長度不對，而其中'
      + '每一格看上去都沒問題。0（預設）：採用節拍自己的值'],
  'cfg.temporal-resample.method': ['',
      '帧率变化所丢弃的信息如何处理。"nearest"（默认）丢帧和复帧（ffmpeg 的'
      + ' `fps`）——最省，但会产生混叠：大比例降帧会抖动，车轮会倒转。"blend'
      + '" 把每个输出帧所落在的两个源帧混合起来（`framerate`）。"average" '
      + '先把每个输出帧时间区间内的源帧做盒式平均再抽取（`tmix` + `fps`）——'
      + '相当于空间上缩小时的面积平均，看起来就是运动模糊，因为它本来就是。'
      + '"motion" 估计运动并合成中间帧（`minterpolate`），这是升帧时唯一有'
      + '用的方式，因为那时根本没有东西可平均。开销依次递增，`motion` 远为'
      + '昂贵',
      '影格率變化所丟棄的資訊如何處理。"nearest"（預設）丟格和複格（ffmpeg '
      + '的 `fps`）——最省，但會產生疊頻：大比例降格會抖動，車輪會倒轉。"'
      + 'blend" 把每個輸出影格所落在的兩個來源影格混合起來（`framerate`）。'
      + '"average" 先把每個輸出影格時間區間內的來源影格做盒式平均再抽取（`'
      + 'tmix` + `fps`）——相當於空間上縮小時的面積平均，看起來就是動態模糊'
      + '，因為它本來就是。"motion" 估計運動並合成中間影格（`minterpolate`'
      + '），這是升格時唯一有用的方式，因為那時根本沒有東西可平均。開銷依次'
      + '遞增，`motion` 遠為昂貴'],
  'cfg.temporal-resample.average_frames': ['',
      '`average` 的窗口长度，以源帧计。0（默认）由比例推导——round(input_fps'
      + ' / output_fps)，也就是一个输出帧所覆盖的区间。调大相当于更长的快门'
      + '，调小则更短',
      '`average` 的視窗長度，以來源影格計。0（預設）由比例推導——round('
      + 'input_fps / output_fps)，也就是一個輸出影格所涵蓋的區間。調大相當'
      + '於更長的快門，調小則更短'],
  'cfg.temporal-resample.stacked': ['',
      'false（默认）：每个节拍一帧，[3, H, W]，输出端口自成一个时钟域，因为'
      + '帧率变化并不是一个节拍进、一个节拍出。true：整段片段装在一个节拍里'
      + '，[T, 3, H, W]——即 `temporal-stack` 构建的形状——于是本阶段是 1:1 '
      + '的，输出端口与输入端口共用时钟，不跨越任何时钟域，也就可以放进反馈'
      + '回路里。这是声明出来的而不是探测出来的：时钟分析在启动时就运行，那'
      + '时还没有任何节拍存在',
      'false（預設）：每個節拍一格，[3, H, W]，輸出埠自成一個時脈域，因為影'
      + '格率變化並不是一個節拍進、一個節拍出。true：整段片段裝在一個節拍裡'
      + '，[T, 3, H, W]——即 `temporal-stack` 建構的形狀——於是本階段是 1:1 '
      + '的，輸出埠與輸入埠共用時脈，不跨越任何時脈域，也就可以放進回饋迴路'
      + '裡。這是宣告出來的而不是探測出來的：時脈分析在啟動時就執行，那時還'
      + '沒有任何節拍存在'],
  'port.temporal-resample.in.frames': ['',
      '平面 u8 RGB TensorBeat：每个节拍 [3, H, W]，或 `stacked` 时的单个 [T'
      + ', 3, H, W] 片段',
      '平面 u8 RGB TensorBeat：每個節拍 [3, H, W]，或 `stacked` 時的單個 [T'
      + ', 3, H, W] 片段'],
  'port.temporal-resample.out.frames': ['',
      '同样的形状，位于 `output_fps`，边带中的帧率已重写。声明的时钟组是流'
      + '式情形下的答案；真正的答案由 oport_clock_group() 决定——堆叠片段是 '
      + '1:1 的，与输入端口共用时钟',
      '同樣的形狀，位於 `output_fps`，邊帶中的影格率已重寫。宣告的時脈組是'
      + '串流情形下的答案；真正的答案由 oport_clock_group() 決定——堆疊片段'
      + '是 1:1 的，與輸入埠共用時脈'],

  // ---- Temporal Slice (visual) ----
  'stage.temporal-slice.name': ['', '时间切片', '時間切片'],
  'stage.temporal-slice.doc': ['',
      '对节拍流做 Python 的 seq[start:end:step]：保留切片指定的节拍，丢弃其'
      + '余，并把保留的每一个原样转发。`start: -1` 接到 save-image 就是写出'
      + '片段的最后一帧。负索引要到 EOS 才能解析，因此代价是要一直持有那么'
      + '多个节拍。',
      '對節拍串流做 Python 的 seq[start:end:step]：保留切片指定的節拍，丟棄'
      + '其餘，並把保留的每一個原樣轉發。`start: -1` 接到 save-image 就是寫'
      + '出片段的最後一格。負索引要到 EOS 才能解析，因此代價是要一直持有那'
      + '麼多個節拍。'],
  'cfg.temporal-slice.start': ['',
      '切片的第一个节拍，Python 风格。负数从末尾往回数——-1 是最后一个节拍，'
      + '-2 是倒数第二个——而这只有等源到达 EOS 才能确定，因此在那之前本阶段'
      + '会一直持有那么多个节拍。不填：从第一个节拍开始',
      '切片的第一個節拍，Python 風格。負數從末尾往回數——-1 是最後一個節拍，'
      + '-2 是倒數第二個——而這只有等來源抵達 EOS 才能確定，因此在那之前本階'
      + '段會一直持有那麼多個節拍。不填：從第一個節拍開始'],
  'cfg.temporal-slice.step': ['',
      '从 `start` 起每隔 step 个节拍保留一个。必须为正：流只能顺序读一次，'
      + '负的步长意味着要发出本阶段早已放走的节拍。不填：1',
      '從 `start` 起每隔 step 個節拍保留一個。必須為正：串流只能循序讀一次'
      + '，負的步長意味著要發出本階段早已放走的節拍。不填：1'],
  'cfg.temporal-slice.sequence': ['',
      '切片索引作用于什么。"beats"（默认）是节拍流——保留一些节拍，丢弃其余'
      + '。"frames" 是单个节拍的最前那个轴：送入一段堆叠片段 [T, 3, H, W]，'
      + '取出选中的那些帧，一个节拍进、一个节拍出。frames 模式什么都不持有'
      + '（T 从节拍本身即可得知，负索引当即就能解析），也不改变节拍速率，因'
      + '此与流模式不同，它可以放进反馈回路里',
      '切片索引作用於什麼。"beats"（預設）是節拍串流——保留一些節拍，丟棄其'
      + '餘。"frames" 是單個節拍的最前那個軸：送入一段堆疊片段 [T, 3, H, W]'
      + '，取出選中的那些影格，一個節拍進、一個節拍出。frames 模式什麼都不'
      + '持有（T 從節拍本身即可得知，負索引當即就能解析），也不改變節拍速率'
      + '，因此與串流模式不同，它可以放進回饋迴路裡'],
  'cfg.temporal-slice.squeeze': ['',
      '仅 frames 模式：当切片恰好选中一帧时，去掉时间轴，发出 [3, H, W]——一'
      + '张静态图——而不是 [1, 3, H, W] 这样的单帧片段。对参考条件模型来说这'
      + '是两种不同的请求，尺寸规则也不同，因此没有安全的默认值，只能做成一'
      + '个开关而不是自动推断。要求对多帧做 squeeze 会得到一次警告，时间轴'
      + '则予以保留',
      '僅 frames 模式：當切片恰好選中一格時，去掉時間軸，發出 [3, H, W]——一'
      + '張靜態圖——而不是 [1, 3, H, W] 這樣的單格片段。對參考條件模型來說這'
      + '是兩種不同的請求，尺寸規則也不同，因此沒有安全的預設值，只能做成一'
      + '個開關而不是自動推斷。要求對多格做 squeeze 會得到一次警告，時間軸'
      + '則予以保留'],
  'cfg.temporal-slice.end': ['',
      '切片最后一个节拍的下一个位置，Python 风格，因此切片是 [start, end)。'
      + '负数从末尾往回数，规则与 `start` 相同。不填：一直运行到源到达 EOS',
      '切片最後一個節拍的下一個位置，Python 風格，因此切片是 [start, end)。'
      + '負數從末尾往回數，規則與 `start` 相同。不填：一直執行到來源抵達 '
      + 'EOS'],
  'port.temporal-slice.in.beats': ['',
      '任意节拍流；一个节拍就是序列中的一个位置',
      '任意節拍串流；一個節拍就是序列中的一個位置'],
  'port.temporal-slice.out.beats': ['',
      '被选中的节拍，原样转发——负载、形状、边带都不变。不做堆叠：N 个节拍的'
      + '切片仍是 N 个节拍，而不是一个多出时间轴的张量（那是 temporal-stack'
      + ' 的事）',
      '被選中的節拍，原樣轉發——負載、形狀、邊帶都不變。不做堆疊：N 個節拍的'
      + '切片仍是 N 個節拍，而不是一個多出時間軸的張量（那是 temporal-stack'
      + ' 的事）'],

  // ---- Temporal Stack (visual) ----
  'stage.temporal-stack.name': ['', '时间堆叠', '時間堆疊'],
  'stage.temporal-stack.doc': ['',
      '把许多时间切片节拍堆叠成一个张量：帧堆成片段，PCM 分块堆成完整波形。'
      + '它是本代码树的流式约定与“需要一整段参考作为条件”的消费者之间的桥梁'
      + '。跨越“节拍速率 -> 分组速率”两个时钟域。',
      '把許多時間切片節拍堆疊成一個張量：影格堆成片段，PCM 分塊堆成完整波形'
      + '。它是本程式碼樹的串流約定與「需要一整段參考作為條件」的消費者之間'
      + '的橋樑。跨越「節拍速率 -> 分組速率」兩個時脈域。'],
  'cfg.temporal-stack.mode': ['',
      '如何堆叠：auto（默认）| video | audio | generic。video 在最前面新增'
      + '一个轴（[3,H,W] x T -> [T,3,H,W]）；audio 在最后一个轴上拼接（[N] '
      + 'x k -> [总 N]，平面的 [2,N] x k -> [2,总 N]）；generic 用 video 的'
      + '规则但不做模态检查。auto 从第一个节拍读出模式——秩为 3 的 u8 是帧，'
      + '秩小于等于 2 的 f32 或任何带 `sample_rate` 的都是音频——并把它锁定'
      + '，因此中途改变形状的生产者会得到一个错误，而不是一个堆到一半的张量',
      '如何堆疊：auto（預設）| video | audio | generic。video 在最前面新增'
      + '一個軸（[3,H,W] x T -> [T,3,H,W]）；audio 在最後一個軸上串接（[N] '
      + 'x k -> [總 N]，平面的 [2,N] x k -> [2,總 N]）；generic 用 video 的'
      + '規則但不做模態檢查。auto 從第一個節拍讀出模式——秩為 3 的 u8 是影格'
      + '，秩小於等於 2 的 f32 或任何帶 `sample_rate` 的都是音訊——並把它鎖'
      + '定，因此中途改變形狀的生產者會得到一個錯誤，而不是一個堆到一半的張'
      + '量'],
  'cfg.temporal-stack.group_size': ['',
      '每发出一组包含多少个节拍。0（默认）一直累积到源结束，然后发出一个节'
      + '拍——这正是单个参考所需要的，同时也让本阶段成为一次性的：发完那个节'
      + '拍它就结束了。要服务多次请求的图必须设置它，否则第一次之后的每次请'
      + '求看到的都是一个已经结束的生产者',
      '每發出一組包含多少個節拍。0（預設）一直累積到來源結束，然後發出一個'
      + '節拍——這正是單個參考所需要的，同時也讓本階段成為一次性的：發完那個'
      + '節拍它就結束了。要服務多次請求的圖必須設定它，否則第一次之後的每次'
      + '請求看到的都是一個已經結束的生產者'],
  'cfg.temporal-stack.overlap': ['',
      '每组保留多少个节拍，好让下一组以它们开头——这段上下文让消费者能看到紧'
      + '接其前的内容。单位是节拍而不是秒，与 group_size 相同：24 fps 下重'
      + '叠 24 就是一秒。必须小于 group_size，此后每次发出前进 (group_size '
      + '- overlap) 个节拍。仅 video 与 generic 分组适用——音频请改用 '
      + 'audio-to-pcm 的 chunk_overlap_s，它以采样点为单位。0（默认）= 各组'
      + '之间不共享任何内容',
      '每組保留多少個節拍，好讓下一組以它們開頭——這段上下文讓消費者能看到緊'
      + '接其前的內容。單位是節拍而不是秒，與 group_size 相同：24 fps 下重'
      + '疊 24 就是一秒。必須小於 group_size，此後每次發出前進 (group_size '
      + '- overlap) 個節拍。僅 video 與 generic 分組適用——音訊請改用 '
      + 'audio-to-pcm 的 chunk_overlap_s，它以取樣點為單位。0（預設）= 各組'
      + '之間不共用任何內容'],
  'cfg.temporal-stack.max_mb': ['',
      '累积器的上限，单位 MB（默认 256）。达到上限时会把已持有的内容发出、'
      + '给出警告并结束该流——因为这一组已经不是当初要的那一组，继续下去只会'
      + '发出一串悄悄变短的分组。这个上限在实践中是空间上的：256 MB 是 171 '
      + '帧 960x544 的 u8 RGB，但 1080p 只有 43 帧、4K 只有 10 帧，而 f32 '
      + '音频要 35 分钟才能填满。请在堆叠之前先把帧缩小，而不是调高这一项',
      '累積器的上限，單位 MB（預設 256）。達到上限時會把已持有的內容發出、'
      + '給出警告並結束該串流——因為這一組已經不是當初要的那一組，繼續下去只'
      + '會發出一串悄悄變短的分組。這個上限在實務上是空間上的：256 MB 是 '
      + '171 格 960x544 的 u8 RGB，但 1080p 只有 43 格、4K 只有 10 格，而 '
      + 'f32 音訊要 35 分鐘才能填滿。請在堆疊之前先把影格縮小，而不是調高這'
      + '一項'],
  'cfg.temporal-stack.sideband': ['',
      '一个 JSON 对象，会在本阶段自己的速率字段之后合并进每个发出分组的边带'
      + '。用来说明关于这一组、而其任何一个节拍的生产者都无从知晓的事情——例'
      + '如 MiniMax-H3 的参考端口会读音频节拍上的 `attach: true`，据此把它'
      + '当作前一个参考的配乐而不是独立参考，而 audio-to-pcm 对这两者一无所'
      + '知。这里给出的键优先于计算得到的键，因此也可以用它来纠正源给错的采'
      + '样率',
      '一個 JSON 物件，會在本階段自己的速率欄位之後合併進每個發出分組的邊帶'
      + '。用來說明關於這一組、而其任何一個節拍的生產者都無從知曉的事情——例'
      + '如 MiniMax-H3 的參考埠會讀音訊節拍上的 `attach: true`，據此把它當'
      + '作前一個參考的配樂而不是獨立參考，而 audio-to-pcm 對這兩者一無所知'
      + '。這裡給出的鍵優先於計算得到的鍵，因此也可以用它來糾正來源給錯的取'
      + '樣率'],
  'cfg.temporal-stack.fps': ['',
      '仅视频：当各帧本身不带帧率时，在堆叠后的节拍上公布的帧率。逐帧节拍只'
      + '有在源知道时才会写出 `fps_num`/`fps_den`，因此本阶段先从那一对推导'
      + '，其次从 `timestamp_us` 的跨度推导，最后才用这一项。0（默认）表示'
      + '不要凭空发明——需要帧率的消费者应当拒绝，而不是拿到一个看似合理的错'
      + '误值',
      '僅視訊：當各影格本身不帶影格率時，在堆疊後的節拍上公布的影格率。逐格'
      + '節拍只有在來源知道時才會寫出 `fps_num`/`fps_den`，因此本階段先從那'
      + '一對推導，其次從 `timestamp_us` 的跨度推導，最後才用這一項。0（預'
      + '設）表示不要憑空發明——需要影格率的消費者應當拒絕，而不是拿到一個看'
      + '似合理的錯誤值'],
  'port.temporal-stack.beats': ['',
      'TensorBeatPayload，每个节拍一个时间切片：平面 u8 RGB [3,H,W] 的帧、'
      + 'f32 PCM 的 [N] 或 [声道数,N] 分块，或 generic 模式下形状一致的任意'
      + '张量',
      'TensorBeatPayload，每個節拍一個時間切片：平面 u8 RGB [3,H,W] 的影格'
      + '、f32 PCM 的 [N] 或 [聲道數,N] 分塊，或 generic 模式下形狀一致的任'
      + '意張量'],
  'port.temporal-stack.stacked': ['',
      '每组一个 TensorBeatPayload：video 与 generic 为 [节拍数, ...]，audio'
      + ' 为 [..., 总和]。边带会按分组重建——`timestamp_us` 取自该组第一个节'
      + '拍，音频给出 `sample_rate` 与 `duration_us`，视频给出解析后的 `fps'
      + '`',
      '每組一個 TensorBeatPayload：video 與 generic 為 [節拍數, ...]，audio'
      + ' 為 [..., 總和]。邊帶會按分組重建——`timestamp_us` 取自該組第一個節'
      + '拍，音訊給出 `sample_rate` 與 `duration_us`，視訊給出解析後的 `fps'
      + '`'],

  // ---- Video Capture (visual) ----
  'stage.video-capture.name': ['', '视频采集', '視訊擷取'],
  'stage.video-capture.doc': ['',
      '源：通过 FFmpeg avfoundation 采集摄像头，每帧发出一个平面 RGB '
      + 'TensorBeat。仅 Apple 平台。无输入端口。',
      '來源：透過 FFmpeg avfoundation 擷取攝影機，每格發出一個平面 RGB '
      + 'TensorBeat。僅 Apple 平台。無輸入埠。'],
  'cfg.video-capture.device_id': ['',
      'avfoundation 视频设备索引（与 device_name 互斥；视频索引与音频分开编'
      + '号）',
      'avfoundation 視訊裝置索引（與 device_name 互斥；視訊索引與音訊分開編'
      + '號）'],
  'cfg.video-capture.device_name': ['',
      'avfoundation 视频设备名称；不区分大小写的子串匹配（与 device_id 互斥'
      + '）',
      'avfoundation 視訊裝置名稱；不區分大小寫的子字串比對（與 device_id 互'
      + '斥）'],
  'cfg.video-capture.width': ['',
      '请求的采集宽度；需与 height 一同设置（avfoundation 的 video_size）。'
      + '0 = 设备默认',
      '請求的擷取寬度；需與 height 一同設定（avfoundation 的 video_size）。'
      + '0 = 裝置預設'],
  'cfg.video-capture.height': ['',
      '请求的采集高度；需与 width 一同设置（avfoundation 的 video_size）。0'
      + ' = 设备默认',
      '請求的擷取高度；需與 width 一同設定（avfoundation 的 video_size）。0'
      + ' = 裝置預設'],
  'cfg.video-capture.framerate': ['',
      '请求的每秒帧数（avfoundation 的 framerate）。0 = 设备默认',
      '請求的每秒影格數（avfoundation 的 framerate）。0 = 裝置預設'],
  'cfg.video-capture.pixel_format': ['',
      '请求的采集像素格式（avfoundation 的 pixel_format），例如 uyvy422 / '
      + 'nv12 / bgr0；留空 = 设备默认。无论如何输出都是 RGB',
      '請求的擷取像素格式（avfoundation 的 pixel_format），例如 uyvy422 / '
      + 'nv12 / bgr0；留空 = 裝置預設。無論如何輸出都是 RGB'],
  'cfg.video-capture.output_dtype': ['',
      '发出的元素类型："u8"（默认）或 "f32"（归一化到 [0,1]）',
      '發出的元素型別："u8"（預設）或 "f32"（正規化到 [0,1]）'],
  'cfg.video-capture.camera_name': ['',
      '复制到每个节拍边带中的标签，好让多摄像头的图能区分来源',
      '複製到每個節拍邊帶中的標籤，好讓多攝影機的圖能區分來源'],
  'cfg.video-capture.reconnect_delay_ms': ['',
      '出错后重新打开前的退避时间（毫秒）',
      '出錯後重新開啟前的退避時間（毫秒）'],
  'cfg.video-capture.oport_depth': ['',
      '输出环形缓冲深度（丢弃最旧）',
      '輸出環形緩衝深度（丟棄最舊）'],
  'port.video-capture.frames': ['',
      '平面 RGB TensorBeat [3,H,W]（U8 或 F32），每采集一帧一个——与 '
      + 'video-to-rgb 发出的负载相同',
      '平面 RGB TensorBeat [3,H,W]（U8 或 F32），每擷取一格一個——與 '
      + 'video-to-rgb 發出的負載相同'],

  // ---- Video → RGB (visual) ----
  'stage.video-to-rgb.name': ['', '视频流 → RGB', '視訊串流 → RGB'],
  'stage.video-to-rgb.doc': ['',
      '把 H.264 的 EncodedSegment 解码（FFmpeg 软解/VideoToolbox）为平面 '
      + 'RGB TensorBeat，每帧一个；可选裁剪与缩放。跨越“分段速率 -> 帧速率”'
      + '两个时钟域。',
      '把 H.264 的 EncodedSegment 解碼（FFmpeg 軟解/VideoToolbox）為平面 '
      + 'RGB TensorBeat，每格一個；可選裁剪與縮放。跨越「分段速率 -> 影格速'
      + '率」兩個時脈域。'],
  'cfg.video-to-rgb.normalize': ['',
      '仅 F32：把字节值除以 255 -> [0,1]',
      '僅 F32：把位元組值除以 255 -> [0,1]'],
  'cfg.video-to-rgb.oport_capacity': ['',
      '输出端口缓冲深度',
      '輸出埠緩衝深度'],
  'cfg.video-to-rgb.hwaccel': ['',
      'auto | videotoolbox | none',
      'auto | videotoolbox | none'],
  'cfg.video-to-rgb.output_dtype': ['', 'f32 | u8', 'f32 | u8'],
  'cfg.video-to-rgb.output_width': ['',
      '缩放目标宽度；需与 output_height 成对设置',
      '縮放目標寬度；需與 output_height 成對設定'],
  'cfg.video-to-rgb.output_height': ['',
      '缩放目标高度；需与 output_width 成对设置',
      '縮放目標高度；需與 output_width 成對設定'],
  'port.video-to-rgb.segments': ['',
      'EncodedSegment：AVCC 格式的 H.264 + extradata',
      'EncodedSegment：AVCC 格式的 H.264 + extradata'],
  'port.video-to-rgb.frames': ['',
      '平面 RGB TensorBeat [3,H,W]（F32 或 U8）',
      '平面 RGB TensorBeat [3,H,W]（F32 或 U8）'],
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

// Merge a message catalogue contributed at runtime, in the same
// [en-us, zh-cn, zh-tw] tuple shape as STRINGS. This is how a
// STAGE-PROVIDED view supplies its own text (see stage-views.js): its
// strings ship inside the view module, next to the stage's C++, instead
// of being copied into the catalogue above. Existing keys are NOT
// overwritten, so a view can never redefine an app string.
export function addStrings(table) {
  if (!table || typeof table !== 'object') { return; }
  for (const [key, row] of Object.entries(table)) {
    if (!Array.isArray(row) || key in STRINGS) { continue; }
    STRINGS[key] = row;
  }
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
