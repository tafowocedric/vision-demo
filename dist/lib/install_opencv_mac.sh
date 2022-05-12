#########################
# INSTALL OPENCV ON MAC #
#########################

# -------------------------------------------------------------------- |
#                       SCRIPT OPTIONS                                 |
# ---------------------------------------------------------------------|
OPENCV_VERSION='3.4.17'       # Version to be installed
OPENCV_CONTRIB='YES'          # Install OpenCV's extra modules (YES/NO)
# -------------------------------------------------------------------- |

# 1. INSTALLATION DIRECTORY "Home"
cd ~

sudo mkdir -p Open_CV && cd Open_CV

# 2. BREW UP TO DATE
brew update
brew install python3 cmake qt5 pkg-config jpeg
brew install eigen tbb libpng libtiff openexr
brew install hdf5

# 3. INSTALL THE LIBRARY
sudo wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
sudo unzip ${OPENCV_VERSION}.zip && sudo rm ${OPENCV_VERSION}.zip
sudo mv opencv-${OPENCV_VERSION} opencv

if [ $OPENCV_CONTRIB = 'YES' ]; then
  sudo wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
  sudo unzip ${OPENCV_VERSION}.zip && sudo rm ${OPENCV_VERSION}.zip
  sudo mv opencv_contrib-${OPENCV_VERSION} opencv_contrib
fi

sudo mkdir -p build && cd build

if [ $OPENCV_CONTRIB = 'NO' ]; then
sudo cmake -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_GENERATE_PKGCONFIG=YES -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DWITH_V4L=ON -DWITH_EIGEN=ON -DWITH_1394=ON -DWITH_GSTREAMER=ON -DWITH_IMGCODEC_HDR=ON -DWITH_IMGCODEC_PXM=ON \
      -DWITH_IMGCODEC_SUNRASTER=ON -DWITH_JASPER=ON -DWITH_JPEG=ON -DWITH_OPENCLAMDBLAS=ON -DWITH_OPENCLAMDFFT=ON \
      -DWITH_OPENEXR=ON -DWITH_PNG=ON -DWITH_PROTOBUF=ON -DWITH_PTHREADS_PF=ON -DWITH_QUIRC=ON -DWITH_TIFF=ON -DWITH_WEBP=ON \
      -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_python3=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_GDAL=ON -DWITH_XINE=ON \
      -DENABLE_PRECOMPILED_HEADERS=OFF .
fi

if [ $OPENCV_CONTRIB = 'YES' ]; then
sudo cmake -DCMAKE_BUILD_TYPE=RELEASE -DOPENCV_GENERATE_PKGCONFIG=YES -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DWITH_V4L=ON -DWITH_EIGEN=ON -DWITH_1394=ON -DWITH_GSTREAMER=ON -DWITH_IMGCODEC_HDR=ON -DWITH_IMGCODEC_PXM=ON \
      -DWITH_IMGCODEC_SUNRASTER=ON -DWITH_JASPER=ON -DWITH_JPEG=ON -DWITH_OPENCLAMDBLAS=ON -DWITH_OPENCLAMDFFT=ON \
      -DWITH_OPENEXR=ON -DWITH_PNG=ON -DWITH_PROTOBUF=ON -DWITH_PTHREADS_PF=ON -DWITH_QUIRC=ON -DWITH_TIFF=ON -DWITH_WEBP=ON \
      -DOPENCV_ENABLE_NONFREE=ON -DBUILD_opencv_python3=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_GDAL=ON -DWITH_XINE=ON \
      -DENABLE_PRECOMPILED_HEADERS=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules .
fi

make

sudo make install

# 4. EXECUTE SOME OPENCV EXAMPLES AND COMPILE A DEMONSTRATION

# To complete this step, please visit 'http://milq.github.io/install-opencv-ubuntu-debian'.