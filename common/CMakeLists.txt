set(LOCAL_SOURCES
  credential-store.cc
  db-log-delegate.cc
  ffmpeg-libraries.cc
  ffmpeg-log-tap.cc
  flex-data.cc
  graph.cc
  host-identity.cc
  host-net.cc
  i18n.cc
  library-handle.cc
  lmdb-cursor.cc
  lmdb-db.cc
  lmdb-env.cc
  lmdb-txn.cc
  media-decode.cc
  media-line.cc
  mini-http.cc
  onvif-soap.cc
  perf-buffer.cc
  segment-publisher.cc
  segment-writer.cc
  session-config.cc
  session-manager.cc
  session-member.cc
  session.cc
  static-file-server.cc
  stdio-ui-delegate.cc
  stdout-log-delegate.cc
  thread-pool.cc
  vertex.cc
  ws-discovery.cc
)
if(APPLE)
  list(APPEND LOCAL_SOURCES credential-cipher.cc)
  list(APPEND LOCAL_SOURCES onvif-refresh.cc)
endif()
set(LOCAL_HEADERS
  credential-store.h
  db-log-delegate.h
  encoded-segment.h
  ffmpeg-libraries.h
  ffmpeg-log-tap.h
  flex-data.h
  graph.h
  host-identity.h
  host-net.h
  i18n.h
  job.h
  library-handle.h
  lmdb-cursor.h
  lmdb-db.h
  lmdb-env.h
  lmdb-txn.h
  media-decode.h
  media-line.h
  mini-http.h
  onvif-soap.h
  oport-policy.h
  perf-buffer.h
  perf-event.h
  segment-publisher.h
  segment-writer.h
  session-config.h
  session-member.h
  session.h
  static-file-server.h
  stdio-ui-delegate.h
  stdout-log-delegate.h
  thread-pool.h
  beat-payload-intf.h
  vertex.h
  vpipe-format.h
  ws-discovery.h
)
if(APPLE)
  list(APPEND LOCAL_HEADERS credential-cipher.h)
  list(APPEND LOCAL_HEADERS onvif-refresh.h)
endif()

list(TRANSFORM LOCAL_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(TRANSFORM LOCAL_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND VPIPE_SOURCES ${LOCAL_SOURCES})
list(APPEND VPIPE_HEADERS ${LOCAL_HEADERS})
set(VPIPE_SOURCES ${VPIPE_SOURCES} PARENT_SCOPE)
set(VPIPE_HEADERS ${VPIPE_HEADERS} PARENT_SCOPE)
