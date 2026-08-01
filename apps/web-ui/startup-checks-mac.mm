// macOS microphone- and camera-permission probes for the web-ui startup
// checks. Uses AVCaptureDevice's TCC authorization status -- a passive query
// that does NOT open the device (so it can't hang or grab the mic/camera),
// the same status the System Settings > Privacy & Security > Microphone /
// Camera panes reflect. Built only on Apple (see CMakeLists); the non-Apple
// fallbacks live in startup-checks.cc.
//
// Not opening the device matters here: an actual camera open on a system
// that has not granted (or has denied) camera access can BLOCK rather than
// fail, which is exactly what a startup check must not do.

#import <AVFoundation/AVFoundation.h>
#include <CoreAudio/CoreAudio.h>

#include "apps/web-ui/startup-checks.h"

#include <cstdlib>
#include <vector>

namespace vpipe::webui {

int
microphone_auth_status()
{
  AVAuthorizationStatus s =
      [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
  switch (s) {
    case AVAuthorizationStatusAuthorized:
      return 1;
    case AVAuthorizationStatusDenied:
    case AVAuthorizationStatusRestricted:
      return 0;
    case AVAuthorizationStatusNotDetermined:
    default:
      return -1;
  }
}

int
audio_input_device_count()
{
  AudioObjectPropertyAddress devices_addr = {
      kAudioHardwarePropertyDevices,
      kAudioObjectPropertyScopeGlobal,
      kAudioObjectPropertyElementMain };
  UInt32 size = 0;
  OSStatus st = AudioObjectGetPropertyDataSize(
      kAudioObjectSystemObject, &devices_addr, 0, nullptr, &size);
  if (st != noErr) { return -1; }
  if (size == 0) { return 0; }

  const int n = static_cast<int>(size / sizeof(AudioDeviceID));
  std::vector<AudioDeviceID> devs(static_cast<size_t>(n));
  st = AudioObjectGetPropertyData(kAudioObjectSystemObject, &devices_addr, 0,
                                  nullptr, &size, devs.data());
  if (st != noErr) { return -1; }

  int inputs = 0;
  for (AudioDeviceID dev : devs) {
    // A device is an input if its input-scope stream configuration has
    // at least one channel.
    AudioObjectPropertyAddress cfg_addr = {
        kAudioDevicePropertyStreamConfiguration,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain };
    UInt32 cfg_size = 0;
    if (AudioObjectGetPropertyDataSize(dev, &cfg_addr, 0, nullptr, &cfg_size)
            != noErr || cfg_size == 0) {
      continue;
    }
    auto* bl = static_cast<AudioBufferList*>(std::malloc(cfg_size));
    if (bl == nullptr) { continue; }
    UInt32 channels = 0;
    if (AudioObjectGetPropertyData(dev, &cfg_addr, 0, nullptr, &cfg_size, bl)
            == noErr) {
      for (UInt32 i = 0; i < bl->mNumberBuffers; ++i) {
        channels += bl->mBuffers[i].mNumberChannels;
      }
    }
    std::free(bl);
    if (channels > 0) { ++inputs; }
  }

  // Recent macOS can report 0 input channels for the built-in mic until
  // microphone permission is granted, which would falsely look like "no
  // device". The system default-input device is reported regardless of
  // permission, so fall back to it before concluding there's nothing.
  if (inputs == 0) {
    AudioObjectPropertyAddress def_addr = {
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain };
    AudioDeviceID def = kAudioObjectUnknown;
    UInt32 def_size = sizeof(def);
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &def_addr, 0,
                                   nullptr, &def_size, &def) == noErr
        && def != kAudioObjectUnknown) {
      inputs = 1;
    }
  }
  return inputs;
}

int
camera_auth_status()
{
  AVAuthorizationStatus s =
      [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
  switch (s) {
    case AVAuthorizationStatusAuthorized:
      return 1;
    case AVAuthorizationStatusDenied:
    case AVAuthorizationStatusRestricted:
      return 0;
    case AVAuthorizationStatusNotDetermined:
    default:
      return -1;
  }
}

int
video_input_device_count()
{
  // AVCaptureDeviceDiscoverySession needs an explicit device-type list, and
  // the interesting types arrived at different releases -- so build it at
  // runtime. `devicesWithMediaType:` would be one call, but it is deprecated
  // and this target builds with -Wall -Werror.
  NSMutableArray* types = [NSMutableArray array];
  [types addObject:AVCaptureDeviceTypeBuiltInWideAngleCamera];
  if (@available(macOS 14.0, *)) {
    // External (USB/Thunderbolt) cameras, iPhone-as-webcam, and the
    // Desk View companion the built-in camera exposes.
    [types addObject:AVCaptureDeviceTypeExternal];
    [types addObject:AVCaptureDeviceTypeContinuityCamera];
    [types addObject:AVCaptureDeviceTypeDeskViewCamera];
  }
  AVCaptureDeviceDiscoverySession* ds =
      [AVCaptureDeviceDiscoverySession
          discoverySessionWithDeviceTypes:types
                                mediaType:AVMediaTypeVideo
                                 position:AVCaptureDevicePositionUnspecified];
  if (ds == nil) { return -1; }
  const int n = static_cast<int>(ds.devices.count);

  // Enumeration does not require camera permission on macOS, but be
  // defensive the same way the audio probe is: if discovery came back empty,
  // ask for the system default video device before reporting "none".
  if (n == 0
      && [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo] != nil) {
    return 1;
  }
  return n;
}

}  // namespace vpipe::webui
