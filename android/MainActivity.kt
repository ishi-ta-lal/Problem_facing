package com.example.detect

import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.min

@Suppress("SameParameterValue")
class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private var originalImageWidth: Int = 640
    private var originalImageHeight: Int = 640

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)

        val modelPath = "smallface.onnx"
        val imagePath = "modi.jpg"

        Log.d("MainActivity", "Loaded model from $modelPath")
        Log.d("MainActivity", "Loaded image from $imagePath")

        val ortEnv = OrtEnvironment.getEnvironment()
        val ortSession = ortEnv.createSession(loadModel(modelPath))

        Log.d("MainActivity", "OrtSession created with input tensor details")

        val inputStream = assets.open(imagePath)
        val detector = ObjectDetection(this)

        val result = detector.detect(inputStream, ortEnv, ortSession)

        if (result != null) {
            Log.d("MainActivity", "Detection completed successfully")

            val img1Shape = Size(result.outputBitmap.width, result.outputBitmap.height)
            val img0Shape = Size(originalImageWidth, originalImageHeight)
            for (box in result.outputBoxes) {
                scaleCoordsLandmarks(img1Shape, box.coordinates, img0Shape)
            }

            // Apply Non-Maximum Suppression
            val finalBoxes = applyNMS(result.outputBoxes, 0.3f)
            Log.d("MainActivity", "Final boxes after NMS: $finalBoxes")

            val outputBitmap = detector.drawBoundingBox(result.outputBitmap, finalBoxes)
            saveResult(outputBitmap, finalBoxes, "output.json")

            Log.d("MainActivity", "Setting processed image to ImageView")
            imageView.setImageBitmap(outputBitmap)

            val jsonFile = File(filesDir, "output.json")
            val jsonString = jsonFile.readText()
            Log.d("MainActivity", "Output JSON: $jsonString")

            val outputImageFile = File(filesDir, "output.png")
            saveBitmap(outputBitmap, outputImageFile)

            Log.d("MainActivity", "JSON file saved at: ${jsonFile.absolutePath}")
            Log.d("MainActivity", "Image file saved at: ${outputImageFile.absolutePath}")

        } else {
            Log.e("MainActivity", "Detection result is null.")
        }

        ortSession.close()
        ortEnv.close()
    }

    private fun loadModel(modelName: String): ByteArray {
        assets.open(modelName).use { inputStream ->
            return inputStream.readBytes()
        }
    }

    private fun saveResult(outputBitmap: Bitmap, outputBoxes: List<BoundingBox>, fileName: String) {
        val jsonArray = JSONArray()
        for (box in outputBoxes) {
            val jsonObject = JSONObject()
            jsonObject.put("coordinates", JSONArray(box.coordinates))
            jsonObject.put("confidence", box.confidence)
            jsonObject.put("landmarks", JSONArray(box.landmarks))
            jsonArray.put(jsonObject)
        }

        val jsonObject = JSONObject()
        jsonObject.put("boxes", jsonArray)

        val jsonFile = File(filesDir, fileName)
        saveJson(jsonObject, jsonFile)

        val outputImageFile = File(filesDir, "output.png")
        saveBitmap(outputBitmap, outputImageFile)

        Log.d("MainActivity", "JSON file saved at: ${jsonFile.absolutePath}")
        Log.d("MainActivity", "Image file saved at: ${outputImageFile.absolutePath}")
    }

    private fun saveJson(jsonObject: JSONObject, file: File) {
        file.writeText(jsonObject.toString())
    }

    private fun saveBitmap(bitmap: Bitmap, file: File) {
        file.outputStream().use {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
    }

    private fun scaleCoordsLandmarks(img1Shape: Size, coordinates: FloatArray, img0Shape: Size) {
        val gain = minOf(img1Shape.width.toFloat() / img0Shape.width, img1Shape.height.toFloat() / img0Shape.height)
        val padW = (img1Shape.width - img0Shape.width * gain) / 2
        val padH = (img1Shape.height - img0Shape.height * gain) / 2

        for (i in coordinates.indices) {
            if (i % 2 == 0) {
                coordinates[i] = (coordinates[i] - padW) / gain
            } else {
                coordinates[i] = (coordinates[i] - padH) / gain
            }
        }
        Log.d("MainActivity", "Coordinates scaled: ${coordinates.joinToString()}")
    }

    private fun applyNMS(boxes: List<BoundingBox>, threshold: Float): List<BoundingBox> {
        val pickedBoxes = mutableListOf<BoundingBox>()
        val x1 = FloatArray(boxes.size)
        val y1 = FloatArray(boxes.size)
        val x2 = FloatArray(boxes.size)
        val y2 = FloatArray(boxes.size)
        val areas = FloatArray(boxes.size)

        // Calculate x1, y1, x2, y2, and areas for each box
        for ((index, box) in boxes.withIndex()) {
            x1[index] = box.coordinates[0]
            y1[index] = box.coordinates[1]
            x2[index] = box.coordinates[2]
            y2[index] = box.coordinates[3]
            areas[index] = (x2[index] - x1[index] + 1) * (y2[index] - y1[index] + 1)
        }

        // Sort indices by confidence in descending order
        val indices = boxes.indices.sortedByDescending { boxes[it].confidence }.toMutableList()

        while (indices.isNotEmpty()) {
            val last = indices.removeAt(indices.size - 1)
            pickedBoxes.add(boxes[last])

            val suppress = mutableListOf<Int>()
            for (i in indices.indices.reversed()) {
                val j = indices[i]
                val xx1 = max(x1[last], x1[j])
                val yy1 = max(y1[last], y1[j])
                val xx2 = min(x2[last], x2[j])
                val yy2 = min(y2[last], y2[j])

                val w = max(0.0f, xx2 - xx1 + 1)
                val h = max(0.0f, yy2 - yy1 + 1)

                val overlap = if (w <= 0 || h <= 0) 0.0f else w * h / (areas[last] + areas[j] - w * h)

                if (overlap > threshold) {
                    suppress.add(j)
                }
            }

            suppress.forEach { indices.remove(it) }
        }

        return pickedBoxes
    }


}