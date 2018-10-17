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
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using ONNX_model_test_app.ONNX_models;
using Windows.Media;
using Windows.Graphics.Imaging;

namespace ONNX_model_test_app.ViewModels
{
    public class MainPageViewModel : ViewModelBase
    {
        #region Webcam pivot
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
        #endregion

        #region Image pivot
        public SoftwareBitmapSource OriginalImage { get { return _OriginalImage; } set { Set(ref _OriginalImage, value); } }
        private SoftwareBitmapSource _OriginalImage;
        public SoftwareBitmapSource ModelOutputImage { get { return _ModelOutputImage; } set { Set(ref _ModelOutputImage, value); } }
        private SoftwareBitmapSource _ModelOutputImage;
        #endregion

        private bool loadingPageDone = false;
        private bool leavingPage = false;
        private bool changingCamera = false;
        private MediaCapture mediaCapture;
        private TinyYoloV2Model tinyYoloV2Model;

        #region Webcam pivot
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
        #endregion

        #region Image pivot
        public async void OpenImageFromFile()
        {
            var picker = new Windows.Storage.Pickers.FileOpenPicker
            {
                ViewMode = Windows.Storage.Pickers.PickerViewMode.Thumbnail,
                SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary
            };
            picker.FileTypeFilter.Add(".jpg");
            picker.FileTypeFilter.Add(".jpeg");
            picker.FileTypeFilter.Add(".png");

            var file = await picker.PickSingleFileAsync();
            if (file == null) // The user cancelled the picking operation
                return;

            SoftwareBitmap bitmap;
            using (var stream = await file.OpenReadAsync())
            {
                // Create the decoder from the stream
                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                // Get the SoftwareBitmap representation of the file
                // convert bitmap to avoid this error: https://stackoverflow.com/questions/36865978/why-i-get-error-like-this-using-setbitmapasync
                bitmap = await decoder.GetSoftwareBitmapAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
            }

            // set the original image
            var bitmapSource = new SoftwareBitmapSource();
            await bitmapSource.SetBitmapAsync(bitmap);
            OriginalImage = bitmapSource;

            // todo continue work here after Windows October update

            //// create the output from the model
            //if (tinyYoloV2Model == null)
            //    await LoadModel();

            //var input = new TinyYoloV2ModelInput() { Image = VideoFrame.CreateWithSoftwareBitmap(bitmap) };
            //var output = await tinyYoloV2Model.EvaluateAsync(input);

            // todo interprete output

            // todo use GPU. see doc: https://docs.microsoft.com/en-us/windows/ai/integrate-model
        }

        private async Task LoadModel()
        {
            // todo check if this has to run on a seperate thread so that the UI threas isn't blocked.
            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/Models/Tiny-YOLOv2.onnx"));
            tinyYoloV2Model = await TinyYoloV2Model.CreateModel(modelFile);
        }
        #endregion

        private async Task ShowErrorMessage(string message)
        {
            await new MessageDialog("Error Occured: " + message).ShowAsync();
        }

        public void OnNavigatingFrom() => leavingPage = true;
    }
}
