package com.example.detect

import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

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

            val outputBitmap = result.outputBitmap
            val finalBoxes = result.outputBoxes

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

        for (box in outputBoxes) {            val jsonObject = JSONObject()

            // Convert coordinates to Double and add to JSONArray
            val coordArray = JSONArray()
            for (coord in box.coordinates) {
                coordArray.put(coord.toDouble())
            }

            // Convert landmarks to Double and add to JSONArray
            val landmarkArray = JSONArray()
            for (landmark in box.landmarks) {
                landmarkArray.put(landmark.toDouble())
            }

            jsonObject.put("coordinates", coordArray)
            jsonObject.put("confidence", box.confidence)
            jsonObject.put("landmarks", landmarkArray)
            jsonObject.put("classNum", box.classNum)

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
}
