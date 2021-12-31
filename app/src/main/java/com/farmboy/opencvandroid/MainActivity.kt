package com.farmboy.opencvandroid

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.util.Log
import android.view.OrientationEventListener
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var screenRotation: Int = 0

    // view
    private val viewFinder by lazy { findViewById<JavaCameraView>(R.id.cameraView) }
    private val orientationText by lazy { findViewById<TextView>(R.id.rotation_tv) }

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

        // Face model
        private const val FACE_DIR = "facelib"
        private const val FACE_MODEL = "haarcascade_frontalface_alt2.xml"
        private const val byteSize = 4096 // buffer size

        // RGB
        private val YELLOW = Scalar(255.0, 255.0, 0.0)
        private val BLUE = Scalar(0.0, 0.0, 255.0)
        private val RED = Scalar(255.0, 0.0, 0.0)
        private val GREEN = Scalar(0.0, 255.0, 0.0)
    }

    private fun checkOpenCV(context: Context) =
        if (OpenCVLoader.initDebug()) {
            loadEvents()
            shortMsg(context, OPENCV_SUCCESSFUL)
            Log.d(TAG, OPENCV_SUCCESSFUL)
        } else {
            shortMsg(context, OPENCV_FAIL)
            Log.d(TAG, OPENCV_FAIL)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.clearFlags(WindowManager.LayoutParams.FLAG_FORCE_NOT_FULLSCREEN)
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            checkOpenCV(this)
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }

    /**
     * Process result from permission request dialog box, has the request
     * been granted? If yes, start Camera. Otherwise display a toast
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
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

        val mOrientationEventListener = object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                // Monitors orientation values to determine the target rotation value
                when (orientation) {
                    in 45..134 -> {
                        orientationText.text = getString(R.string.n_270_degree)
                    }
                    in 135..224 -> {
                        orientationText.text = getString(R.string.n_180_degree)
                    }
                    in 225..314 -> {
                        orientationText.text = getString(R.string.n_90_degree)
                    }
                    else -> {
                        orientationText.text = getString(R.string.n_0_degree)
                    }
                }
                screenRotation = orientationText.text.toString().toInt()
            }
        }
        if (mOrientationEventListener.canDetectOrientation()) {
            mOrientationEventListener.enable()
        } else {
            mOrientationEventListener.disable()
        }

        viewFinder.visibility = SurfaceView.VISIBLE
        viewFinder.setCameraIndex(
            CameraCharacteristics.LENS_FACING_FRONT
        )
        viewFinder.setCvCameraViewListener(this)

        checkOpenCVManager()
    }

    private fun checkOpenCVManager() {
        handleFaceDetect()
        if (faceDetector!!.empty()) {
            faceDetector = null
        } else {
            faceDir.delete()
        }
        viewFinder.enableView()
    }

    // face library
    var faceDetector: CascadeClassifier? = null
    lateinit var faceDir: File
    var imageRatio = 0.0 // scale down ratio

    private fun handleFaceDetect() {
        try {
            val modelInputStream =
                resources.openRawResource(
                    R.raw.haarcascade_frontalface_alt2
                )

            // create a temp directory
            faceDir = getDir(FACE_DIR, Context.MODE_PRIVATE)

            // create a model file
            val faceModel = File(faceDir, FACE_MODEL)

            if (!faceModel.exists()) { // copy model
                // copy model to new face library
                val modelOutputStream = FileOutputStream(faceModel)

                val buffer = ByteArray(byteSize)
                var byteRead = modelInputStream.read(buffer)
                while (byteRead != -1) {
                    modelOutputStream.write(buffer, 0, byteRead)
                    byteRead = modelInputStream.read(buffer)
                }

                modelInputStream.close()
                modelOutputStream.close()
            }

            faceDetector = CascadeClassifier(faceModel.absolutePath)
        } catch (e: IOException) {
            Log.e(TAG, "Error loading cascade face model...$e")
        }
    }

    override fun onResume() {
        super.onResume()
//        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, cvBaseLoaderCallback)
    }

    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    // image storage
    lateinit var imageMat: Mat
    lateinit var grayMat: Mat

    override fun onCameraViewStarted(width: Int, height: Int) {
        imageMat = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        imageMat.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        imageMat = inputFrame!!.rgba()
//        grayMat = inputFrame.gray()
        grayMat = get480Image(inputFrame.gray())
//        imageRatio = 1.0

        // detect face rectangle
        drawFaceRectangle()
        return imageMat
    }

    fun ratioTo480(src: Size): Double {
        val w = src.width
        val h = src.height
        val heightMax = 480
        var ratio: Double = 0.0

        if (w > h) {
            if (w < heightMax) return 1.0
            ratio = heightMax / w
        } else {
            if (h < heightMax) return 1.0
            ratio = heightMax / h
        }

        return ratio
    }

    fun get480Image(src: Mat): Mat {
        val imageSize = Size(src.width().toDouble(), src.height().toDouble())
        imageRatio = ratioTo480(imageSize)

        if (imageRatio.equals(1.0)) return src

        // Downsize image
        val dstSize = Size(imageSize.width * imageRatio, imageSize.height * imageRatio)
        val dst = Mat()
        Imgproc.resize(src, dst, dstSize)

        // Check rotation
        when (screenRotation) {
            0-> {
                Core.rotate(dst, dst, Core.ROTATE_90_CLOCKWISE)
                //mirror
                Core.flip(dst, dst, 1)
            }
            180-> {
                Core.rotate(dst, dst, Core.ROTATE_90_COUNTERCLOCKWISE)
//                Core.flip(dst, dst, 1)
            }
            270-> {
                Core.flip(dst, dst, 0)
            }
        }
        return dst
    }

    private fun drawFaceRectangle() {
        val faceRects = MatOfRect()
        faceDetector!!.detectMultiScale(
            grayMat,
            faceRects)

        val scrW = imageMat.width().toDouble()
        val scrH = imageMat.height().toDouble()

        for (rect in faceRects.toArray()) {
            var x = rect.x.toDouble()
            var y = rect.y.toDouble()
            var w = 0.0
            var h = 0.0
            var rw = rect.width.toDouble() // rectangle width
            var rh = rect.height.toDouble() // rectangle height

            if (imageRatio.equals(1.0)) {
                w = x + rw
                h = y + rh
            } else {
                x /= imageRatio
                y /= imageRatio
                rw /= imageRatio
                rh /= imageRatio
                w = x + rw
                h = y + rh
            }

            when (screenRotation) {
                90-> {
                    rectFace(x, y, w, h, RED)
                    drawDot(x, y, GREEN)
                }
                0-> {
                    rectFace(y, x, h, w, RED)
                    drawDot(y, x, GREEN)
                }
                180-> {
                    Log.d(TAG,"x: $x -- y: $y :: sW: $scrW, sH: $scrH")

//                    rectFace(y, x, h, w, RED)
//                    drawDot(y, x, GREEN)

                    // fix height
                    val yFix = scrW - y
                    val hFix = yFix - rh

                    rectFace(yFix, x, hFix, w, YELLOW)
                    drawDot(yFix, x, BLUE)
                }
                270-> {
//                    rectFace(x, y, w, h, RED)
//                    drawDot(x, y, GREEN)

                    // fix height
                    val yFix = scrH - y
                    val hFix = yFix - rh

                    rectFace(x, yFix, w, hFix, YELLOW)
                    drawDot(x, yFix, BLUE)
                }
            }
        }
    }

    /* Draw rectangle of the face:
    x: x-coor of upper corner
    y: y-coor of upper corner
    w: x-coor of opposite corner
    h: y-coor of opposite corner
    color: RGB
 */
    fun rectFace(x: Double, y: Double, w: Double, h: Double, color:Scalar) {
        Imgproc.rectangle(
            imageMat, // image
            Point(x, y), // upper corner
            Point(w, h),  // opposite corner
            color  // RGB
        )
    }

    /*
    Draw a dot:
        x: x-coor of center
        y: y-coor of center
        color: RGB
 */
    fun drawDot(x: Double, y:Double, color:Scalar) {
        Imgproc.circle(
            imageMat, // image
            Point(x, y),  // center
            4, // radius
            color, // RGB
            -1, // thickness: -1 = filled in
            8 // line type
        )
    }

    override fun onPause() {
        super.onPause()
        viewFinder?.let { viewFinder.disableView() }
    }

    override fun onDestroy() {
        super.onDestroy()
        viewFinder?.let { viewFinder.disableView() }
        if (faceDir.exists()) faceDir.delete()
    }


}