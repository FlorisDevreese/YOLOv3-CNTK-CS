using ONNX_model_test_app.ViewModels;
using System;
using System.Threading.Tasks;
using Windows.Media.Capture;
using Windows.UI.Core;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Navigation;

namespace ONNX_model_test_app.Views
{
    public sealed partial class MainPage : Page
    {
        public static MainPage Instance { get; private set; }
        public MainPageViewModel viewModel { get { return this.DataContext as MainPageViewModel; } }

        public MainPage()
        {
            InitializeComponent();
            NavigationCacheMode = Windows.UI.Xaml.Navigation.NavigationCacheMode.Enabled;
            Instance = this;
        }

        public async Task SetSourceOfCaptureElementAsync(MediaCapture mediaCapture)
        {
            CameraPreviewElement.Source = mediaCapture;
            await CameraPreviewElement.Source.StartPreviewAsync();
        }

        /// <summary>
        /// Called just before a page is no longer the active page in a frame. 
        /// </summary>
        protected override void OnNavigatingFrom(NavigatingCancelEventArgs e)
        {
            viewModel.OnNavigatingFrom();
            base.OnNavigatingFrom(e);
        }

    }
}