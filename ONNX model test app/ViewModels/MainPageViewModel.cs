using System.Collections.Generic;
using System;
using System.Linq;
using System.Threading.Tasks;
using Template10.Mvvm;
using Template10.Services.NavigationService;
using Windows.UI.Xaml.Navigation;
using Windows.Storage;
using Windows.Devices.Enumeration;
using Windows.Media.Capture;
using Windows.Media.MediaProperties;
using System.Collections.ObjectModel;
using ONNX_model_test_app.Models;
using ONNX_model_test_app.Views;
using Windows.UI.Popups;
using Windows.UI.Xaml;
using Windows.UI.Core;

namespace ONNX_model_test_app.ViewModels
{
    public class MainPageViewModel : ViewModelBase
    {
        public DeviceInformationCollection Devices { get { return _Devices; } set { Set(ref _Devices, value); } }
        private DeviceInformationCollection _Devices;
        public object SelectedDevice { get { return _SelectedDevice; } set { Set(ref _SelectedDevice, value as DeviceInformation); CameraDeviceChanged(); } }
        private DeviceInformation _SelectedDevice;
        public ObservableCollection<ResolutionWrapper> Resolutions { get { return _Resolutions; } set { Set(ref _Resolutions, value); } }
        private ObservableCollection<ResolutionWrapper> _Resolutions = new ObservableCollection<ResolutionWrapper>();
        public object SelectedResolution { get { return _SelectedResolution; } set { Set(ref _SelectedResolution, value as ResolutionWrapper); } }
        private ResolutionWrapper _SelectedResolution;
        public bool CameraInitialised { get { return _CameraInitialised; } set { Set(ref _CameraInitialised, value); } }
        private bool _CameraInitialised = false;

        private bool loadingPageDone = false;
        private bool leavingPage = false;
        private bool changingCamera = false;
        private MediaCapture mediaCapture;

        public async void LoadSettings()
        {
            loadingPageDone = false;
            mediaCapture = new MediaCapture();

            Devices = await DeviceInformation.FindAllAsync(DeviceClass.VideoCapture);
            SelectedDevice = Devices.Last();

            if (SelectedDevice != null)
            {
                var settings = new MediaCaptureInitializationSettings
                {
                    AudioDeviceId = "",
                    VideoDeviceId = ((DeviceInformation)SelectedDevice).Id,
                    StreamingCaptureMode = StreamingCaptureMode.Video
                };
                await mediaCapture.InitializeAsync(settings);

                UpdateResolutions(mediaCapture.VideoDeviceController.GetAvailableMediaStreamProperties(MediaStreamType.Photo));

                if (SelectedResolution != null)
                {
                    await mediaCapture.VideoDeviceController.SetMediaStreamPropertiesAsync(MediaStreamType.Photo, ((ResolutionWrapper)SelectedResolution).VideoProperties);

                    await MainPage.Instance.SetSourceOfCaptureElementAsync(mediaCapture);

                    CameraInitialised = true;
                }
                else
                    await ShowErrorMessage("No resolutions found");
            }
            else
                await ShowErrorMessage("No camera devices found");

            loadingPageDone = true;
        }

        private async void CameraDeviceChanged()
        {
            if (loadingPageDone && !leavingPage && !changingCamera)
            {
                changingCamera = true;

                await mediaCapture.StopPreviewAsync();
                mediaCapture.Dispose();
                mediaCapture = new MediaCapture();

                // set Resolution posibilities
                var settings = new MediaCaptureInitializationSettings
                {
                    AudioDeviceId = "",
                    VideoDeviceId = ((DeviceInformation)SelectedDevice).Id,
                    StreamingCaptureMode = StreamingCaptureMode.Video
                };

                await mediaCapture.InitializeAsync(settings);

                UpdateResolutions(mediaCapture.VideoDeviceController.GetAvailableMediaStreamProperties(MediaStreamType.Photo));

                if (SelectedResolution != null)
                {
                    await mediaCapture.VideoDeviceController.SetMediaStreamPropertiesAsync(MediaStreamType.Photo, ((ResolutionWrapper)SelectedResolution).VideoProperties);

                    await MainPage.Instance.SetSourceOfCaptureElementAsync(mediaCapture);

                    CameraInitialised = true;
                }
                else
                    await ShowErrorMessage("No resolutions found");

                changingCamera = false;
            }
        }

        /// <summary>
        ///  Updates the Resolutions list, and sets the heighest Resolution as the SelectedResolution
        /// </summary>
        private void UpdateResolutions(IReadOnlyList<IMediaEncodingProperties> videoEncodingProperties)
        {
            Resolutions.Clear();
            SelectedResolution = null;

            uint maxPixelheight = 0;
            foreach (VideoEncodingProperties videoProperties in videoEncodingProperties)
            {
                ResolutionWrapper resolutionWrapper = new ResolutionWrapper { Text = videoProperties.Height + " x " + videoProperties.Width, VideoProperties = videoProperties };
                Resolutions.Add(resolutionWrapper);

                // select the heighest resolution
                if (maxPixelheight < videoProperties.Height)
                {
                    maxPixelheight = videoProperties.Height;
                    SelectedResolution = resolutionWrapper;
                }
            }
        }

        private async Task ShowErrorMessage(string message)
        {
            await new MessageDialog("Error Occured: " + message).ShowAsync();
        }

        public void OnNavigatingFrom() => leavingPage = true;
    }
}
