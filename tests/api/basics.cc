#include "minitest.h"
#include "vpipe/vpipe.h"
#include <iostream>

using namespace std;

TEST(basic, session) {

  cout << vpipe::vpipe_version() << endl;

  auto s = vpipe::SessionManager::get().create_session();
  auto cnt = vpipe::SessionManager::get().num_sessions();
  cout << "see " << cnt << " session." << endl;
  EXPECT_TRUE(cnt == 1);
  vpipe::SessionManager::get().destroy_session(s);

}

