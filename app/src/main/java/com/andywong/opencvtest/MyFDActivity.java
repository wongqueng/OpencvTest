package com.andywong.opencvtest;

import android.content.Context;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import static org.opencv.imgproc.Imgproc.rectangle;

public class MyFDActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    JavaCameraView myjcv;
    Mat mRgba, mTmp;
    Button bt;
    int index = 0;
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private Size mSize0;
    private Mat mIntermediateMat;
    private MatOfInt mChannels[];
    private MatOfInt mHistSize;
    private int mHistSizeNum = 25;
    private Mat mMat0;
    private float[] mBuff;
    private MatOfFloat mRanges;
    private Point mP1;
    private Point mP2;
    private Scalar mColorsRGB[];
    private Scalar mColorsHue[];
    private Scalar mWhilte;
    private Mat mSepiaKernel;
    private CascadeClassifier cascadeClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_my_fd);
        myjcv = (JavaCameraView) findViewById(R.id.myfd_surface_view);
        bt = (Button) findViewById(R.id.bt);
        bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                index++;
            }
        });
        myjcv.setCvCameraViewListener(this);
        OpenCVLoader.initDebug();
    }


    @Override
    protected void onResume() {
        super.onResume();
        myjcv.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        myjcv.enableView();
    }

    @Override
    protected void onPause() {
        super.onPause();
        myjcv.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        myjcv.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mTmp = new Mat(height, width, CvType.CV_8UC4);
        File mCascadeFile = null;
        try {

            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            if (!mCascadeFile.exists()) {
                InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                FileOutputStream os = new FileOutputStream(mCascadeFile);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        ;
        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        mIntermediateMat = new Mat();
        mSize0 = new Size();
        mChannels = new MatOfInt[]{new MatOfInt(0), new MatOfInt(1), new MatOfInt(2)};
        mBuff = new float[mHistSizeNum];
        mHistSize = new MatOfInt(mHistSizeNum);
        mRanges = new MatOfFloat(0f, 256f);
        mMat0 = new Mat();
        mColorsRGB = new Scalar[]{new Scalar(200, 0, 0, 255), new Scalar(0, 200, 0, 255), new Scalar(0, 0, 200, 255)};
        mColorsHue = new Scalar[]{
                new Scalar(255, 0, 0, 255), new Scalar(255, 60, 0, 255), new Scalar(255, 120, 0, 255), new Scalar(255, 180, 0, 255), new Scalar(255, 240, 0, 255),
                new Scalar(215, 213, 0, 255), new Scalar(150, 255, 0, 255), new Scalar(85, 255, 0, 255), new Scalar(20, 255, 0, 255), new Scalar(0, 255, 30, 255),
                new Scalar(0, 255, 85, 255), new Scalar(0, 255, 150, 255), new Scalar(0, 255, 215, 255), new Scalar(0, 234, 255, 255), new Scalar(0, 170, 255, 255),
                new Scalar(0, 120, 255, 255), new Scalar(0, 60, 255, 255), new Scalar(0, 0, 255, 255), new Scalar(64, 0, 255, 255), new Scalar(120, 0, 255, 255),
                new Scalar(180, 0, 255, 255), new Scalar(255, 0, 255, 255), new Scalar(255, 0, 215, 255), new Scalar(255, 0, 85, 255), new Scalar(255, 0, 0, 255)
        };
        mWhilte = Scalar.all(255);
        mP1 = new Point();
        mP2 = new Point();

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mTmp.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Size sizeRgba = mRgba.size();
        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;
        Mat rgbaInnerWindow;

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;

        switch (index % 8) {
            case 1:
                //灰化处理
                Imgproc.cvtColor(inputFrame.gray(), mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case 2:
                //Canny边缘检测
                mRgba = inputFrame.rgba();
                Imgproc.Canny(inputFrame.gray(), mTmp, 80, 100);
                Imgproc.cvtColor(mTmp, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
                break;
            case 3:
                //Hist直方图计算
                Mat hist = new Mat();
                int thikness = (int) (sizeRgba.width / (mHistSizeNum + 10) / 5);
                if (thikness > 5) thikness = 5;
                int offset = (int) ((sizeRgba.width - (5 * mHistSizeNum + 4 * 10) * thikness) / 2);

                // RGB
                for (int c = 0; c < 3; c++) {
                    Imgproc.calcHist(Arrays.asList(mRgba), mChannels[c], mMat0, hist, mHistSize, mRanges);
                    Core.normalize(hist, hist, sizeRgba.height / 2, 0, Core.NORM_INF);
                    hist.get(0, 0, mBuff);
                    for (int h = 0; h < mHistSizeNum; h++) {
                        mP1.x = mP2.x = offset + (c * (mHistSizeNum + 10) + h) * thikness;
                        mP1.y = sizeRgba.height - 1;
                        mP2.y = mP1.y - 2 - (int) mBuff[h];
                        Imgproc.line(mRgba, mP1, mP2, mColorsRGB[c], thikness);
                    }
                }
                // Value and Hue
                Imgproc.cvtColor(mRgba, mTmp, Imgproc.COLOR_RGB2HSV_FULL);
                // Value
                Imgproc.calcHist(Arrays.asList(mTmp), mChannels[2], mMat0, hist, mHistSize, mRanges);
                Core.normalize(hist, hist, sizeRgba.height / 2, 0, Core.NORM_INF);
                hist.get(0, 0, mBuff);
                for (int h = 0; h < mHistSizeNum; h++) {
                    mP1.x = mP2.x = offset + (3 * (mHistSizeNum + 10) + h) * thikness;
                    mP1.y = sizeRgba.height - 1;
                    mP2.y = mP1.y - 2 - (int) mBuff[h];
                    Imgproc.line(mRgba, mP1, mP2, mWhilte, thikness);
                }
                break;
            case 4:
                //Sobel边缘检测
                Mat gray = inputFrame.gray();
                Mat grayInnerWindow = gray.submat(top, top + height, left, left + width);
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Imgproc.Sobel(grayInnerWindow, mIntermediateMat, CvType.CV_8U, 1, 1);
                Core.convertScaleAbs(mIntermediateMat, mIntermediateMat, 10, 0);
                Imgproc.cvtColor(mIntermediateMat, rgbaInnerWindow, Imgproc.COLOR_GRAY2BGRA, 4);
                grayInnerWindow.release();
                rgbaInnerWindow.release();
                break;
            case 5:
                //SEPIA(色调变换)
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Core.transform(rgbaInnerWindow, rgbaInnerWindow, mSepiaKernel);
                rgbaInnerWindow.release();
                break;
            case 6:
                //ZOOM放大镜
                Mat zoomCorner = mRgba.submat(0, rows / 2 - rows / 10, 0, cols / 2 - cols / 10);
                Mat mZoomWindow = mRgba.submat(rows / 2 - 9 * rows / 100, rows / 2 + 9 * rows / 100, cols / 2 - 9 * cols / 100, cols / 2 + 9 * cols / 100);
                Imgproc.resize(mZoomWindow, zoomCorner, zoomCorner.size());
                Size wsize = mZoomWindow.size();
                Imgproc.rectangle(mZoomWindow, new Point(1, 1), new Point(wsize.width - 2, wsize.height - 2), new Scalar(255, 0, 0, 255), 2);
                zoomCorner.release();
                mZoomWindow.release();
                break;
            case 7:
                //PIXELIZE像素化
                rgbaInnerWindow = mRgba.submat(top, top + height, left, left + width);
                Imgproc.resize(rgbaInnerWindow, mIntermediateMat, mSize0, 0.1, 0.1, Imgproc.INTER_NEAREST);
                Imgproc.resize(mIntermediateMat, rgbaInnerWindow, rgbaInnerWindow.size(), 0., 0., Imgproc.INTER_NEAREST);
                rgbaInnerWindow.release();
                break;
            default:
                //显示原图
                mRgba = inputFrame.rgba();
                mTmp = inputFrame.gray();
                MatOfRect facedetect = new MatOfRect();
                cascadeClassifier.detectMultiScale(mRgba, facedetect);
                Rect[] facesArray = facedetect.toArray();
                for (int i = 0; i < facesArray.length; i++)
                    rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
                break;
        }
        //返回处理后的结果数据
        return mRgba;
    }

}
