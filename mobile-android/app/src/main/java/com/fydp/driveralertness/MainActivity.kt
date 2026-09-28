package com.fydp.driveralertness

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.os.Bundle
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.ArrayDeque
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min

class MainActivity : ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var state: TextView
    private lateinit var alert: TextView
    private lateinit var confidence: TextView
    private lateinit var analysisExecutor: ExecutorService
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private val behaviorWindow = ArrayDeque<String>(12)
    private val classNames = arrayOf(
        "Drinking", "Eating", "Normal Focused", "Normal Non-Focused",
        "Sleeping", "Using Phone", "Yawning/Drowsiness"
    )
    private val unsafeBehaviors = setOf(
        "Drinking", "Eating", "Normal Non-Focused", "Sleeping",
        "Using Phone", "Yawning/Drowsiness"
    )

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else state.text = "Camera permission required"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        analysisExecutor = Executors.newSingleThreadExecutor()
        setContentView(buildScreen())
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun buildScreen(): LinearLayout {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 20, 24, 20)
            setBackgroundColor(Color.rgb(16, 21, 27))
        }
        val title = TextView(this).apply {
            text = "DRIVER / ALERTNESS\nCamera-based safety monitor"
            setTextColor(Color.WHITE)
            textSize = 22f
            setPadding(0, 0, 0, 16)
        }
        root.addView(title)
        preview = PreviewView(this)
        root.addView(preview, LinearLayout.LayoutParams(-1, 0, 1f))
        state = infoText("Normal Focused")
        root.addView(state)
        confidence = infoText("Confidence: --")
        confidence.textSize = 13f
        confidence.setTextColor(Color.LTGRAY)
        root.addView(confidence)
        alert = infoText("Monitoring inactive")
        root.addView(alert)
        val note = infoText("Live model: best.onnx")
        note.textSize = 12f
        note.setTextColor(Color.LTGRAY)
        root.addView(note)
        return root
    }

    private fun infoText(value: String) = TextView(this).apply {
        text = value
        textSize = 17f
        gravity = Gravity.CENTER_VERTICAL
        setTextColor(Color.WHITE)
        setPadding(0, 14, 0, 14)
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val cameraPreview = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor) { image -> analyze(image) } }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, cameraPreview, analysis)
            state.text = "Normal Focused"
            alert.text = "Loading model..."
            loadModel()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun loadModel() {
        analysisExecutor.execute {
            try {
                environment = OrtEnvironment.getEnvironment()
                val model = assets.open("best.onnx").use { it.readBytes() }
                session = environment!!.createSession(model)
                runOnUiThread { alert.text = "Camera ready" }
            } catch (error: Exception) {
                runOnUiThread { alert.text = "Model error: ${error.message}" }
            }
        }
    }

    private fun analyze(image: ImageProxy) {
        var bitmap: Bitmap? = null
        try {
            val currentSession = session ?: return
            bitmap = image.toBitmap() ?: return
            val input = preprocess(bitmap)
            val inputName = currentSession.inputNames.iterator().next()
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, 3, 640, 640)).use { tensor ->
                currentSession.run(mapOf(inputName to tensor)).use { result ->
                    val output = result[0].value as Array<Array<FloatArray>>
                    val detection = output[0]
                        .filter { it[4] >= 0.20f }
                        .maxByOrNull { it[4] }
                    val behavior = detection?.let { classNames[it[5].toInt().coerceIn(classNames.indices)] } ?: "Normal Focused"
                    updateBehavior(behavior, detection?.get(4) ?: 0f)
                }
            }
        } catch (_: Exception) {
            runOnUiThread { alert.text = "Inference unavailable" }
        } finally {
            bitmap?.recycle()
            image.close()
        }
    }

    private fun preprocess(bitmap: Bitmap): FloatArray {
        val scale = min(640f / bitmap.width, 640f / bitmap.height)
        val resized = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(resized)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val width = (bitmap.width * scale).toInt()
        val height = (bitmap.height * scale).toInt()
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        canvas.drawBitmap(scaled, ((640 - width) / 2).toFloat(), ((640 - height) / 2).toFloat(), null)
        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        val output = FloatArray(3 * 640 * 640)
        for (index in pixels.indices) {
            val pixel = pixels[index]
            output[index] = ((pixel shr 16) and 255) / 255f
            output[640 * 640 + index] = ((pixel shr 8) and 255) / 255f
            output[2 * 640 * 640 + index] = (pixel and 255) / 255f
        }
        scaled.recycle()
        resized.recycle()
        return output
    }

    private fun updateBehavior(behavior: String, score: Float) {
        behaviorWindow.addLast(behavior)
        if (behaviorWindow.size > 12) behaviorWindow.removeFirst()
        val unsafeCount = behaviorWindow.count { it in unsafeBehaviors }
        val triggered = behavior in unsafeBehaviors && unsafeCount >= 7
        runOnUiThread {
            state.text = behavior
            confidence.text = "Confidence: ${(score.coerceIn(0f, 1f) * 100).toInt()}%"
            alert.text = if (triggered) "WARNING\nSustained $behavior detected" else "No active warning"
        }
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        if (format != ImageFormat.YUV_420_888) return null
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val pixels = IntArray(width * height)
        for (row in 0 until height) {
            val yRow = row * yPlane.rowStride
            val uvRow = (row / 2) * uPlane.rowStride
            for (column in 0 until width) {
                val yValue = (yBuffer.get(yRow + column * yPlane.pixelStride).toInt() and 255) - 16
                val uvColumn = (column / 2) * uPlane.pixelStride
                val uValue = (uBuffer.get(uvRow + uvColumn).toInt() and 255) - 128
                val vValue = (vBuffer.get((row / 2) * vPlane.rowStride + (column / 2) * vPlane.pixelStride).toInt() and 255) - 128
                val red = (1.164f * yValue + 1.596f * vValue).toInt().coerceIn(0, 255)
                val green = (1.164f * yValue - 0.392f * uValue - 0.813f * vValue).toInt().coerceIn(0, 255)
                val blue = (1.164f * yValue + 2.017f * uValue).toInt().coerceIn(0, 255)
                pixels[row * width + column] = Color.rgb(red, green, blue)
            }
        }
        val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        if (imageInfo.rotationDegrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also { bitmap.recycle() }
    }

    override fun onDestroy() {
        session?.close()
        environment?.close()
        analysisExecutor.shutdown()
        super.onDestroy()
    }
}
