# Each application is a self-contained executable target living in
# its own subdirectory under apps/. Add new apps here.
add_subdirectory(db-log-reader)
add_subdirectory(web-ui)
add_subdirectory(vpipe)
# macos-app wraps the two binaries above, so it comes last: its bundling
# target depends on both.
add_subdirectory(macos-app)
