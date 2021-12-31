package com.farmboy.opencvandroid

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    // view
    private val viewFinder by lazy { findViewById<JavaCameraView>(R.id.cameraView) }

    companion object {

        val TAG = "MYLOG " + MainActivity::class.java.simpleName

        fun shortMsg(context: Context, s: String) =
            Toast.makeText(context, s, Toast.LENGTH_SHORT).show()

        // messages:
        private const val OPENCV_SUCCESSFUL = "OpenCV Loaded Successfully!"
        private const val OPENCV_FAIL = "Could not load OpenCV!!!"
        private const val OPENCV_PROBLEM = "There's a problem in OpenCV."
        private const val PERMISSION_NOT_GRANTED = "Permissions not granted by the user."

        // Permission vars:
        private const val REQUEST_CODE_PERMISSIONS = 111
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_FINE_LOCATION
        )
    }

    private fun checkOpenCV(context: Context) =
        if (OpenCVLoader.initDebug()) {
            loadEvents()
            shortMsg(context, OPENCV_SUCCESSFUL)
            Log.d(TAG, OPENCV_SUCCESSFUL)
        }
        else {
            shortMsg(context, OPENCV_FAIL)
            Log.d(TAG, OPENCV_FAIL)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.clearFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN)
        window.setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            checkOpenCV(this)
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS )
        }
    }

    /**
     * Process result from permission request dialog box, has the request
     * been granted? If yes, start Camera. Otherwise display a toast
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                checkOpenCV(this)
            } else {
                shortMsg(this, PERMISSION_NOT_GRANTED)
                finish()
            }
        }
    }

    private fun loadEvents() {
        viewFinder.visibility = SurfaceView.VISIBLE
        viewFinder.setCameraIndex(
            CameraCharacteristics.LENS_FACING_FRONT)
        viewFinder.setCvCameraViewListener(this)

        checkOpenCVManager()
    }

    // openCV callback
    lateinit var cvBaseLoaderCallback: BaseLoaderCallback

    private fun checkOpenCVManager() {
        cvBaseLoaderCallback = object : BaseLoaderCallback(this) {
            override fun onManagerConnected(status: Int) {

                when (status) {
                    SUCCESS -> {
                        Log.d(TAG, OPENCV_SUCCESSFUL)
                        shortMsg(this@MainActivity, OPENCV_SUCCESSFUL)
                        viewFinder.enableView()
                    }

                    else -> super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, cvBaseLoaderCallback)
    }

    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    // image storage
    lateinit var imageMat: Mat

    override fun onCameraViewStarted(width: Int, height: Int) {
        imageMat = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        imageMat.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
            imageMat = inputFrame!!.rgba()
            return imageMat
    }
    override fun onPause() {
        super.onPause()
        viewFinder?.let { viewFinder.disableView() }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewFinder?.let { viewFinder.disableView() }
    }
}