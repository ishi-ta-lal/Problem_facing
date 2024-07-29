package com.example.frs


import ai.onnxruntime.OrtEnvironment
import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView

    @SuppressLint("MissingInflatedId")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)

        // Initialize OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.d("MainActivity", "OpenCV loaded successfully")
        } else {
            Log.e("MainActivity", "Failed to load OpenCV")
        }

        val recognitionModelPath = "recogmodel.onnx"
        val imagePath = "modi.jpg"

        Log.d("MainActivity", "Loaded recognition model from $recognitionModelPath")
        Log.d("MainActivity", "Loaded image from $imagePath")

        val inputStream = assets.open(imagePath)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        imageView.setImageBitmap(bitmap)

        val ortEnv = OrtEnvironment.getEnvironment()
        val recognitionOrtSession = ortEnv.createSession(loadModel(recognitionModelPath))

        val recognition = Recognition(recognitionOrtSession, ortEnv)

        // Initialize Recognition and get embeddings
        val bbox = listOf(73f, 42f, 144f, 133f)
        val landmarks = listOf(92f, 73f, 122f, 73f, 108f, 88f, 95f, 105f, 123f, 104f)

        val embeddings = recognition.getEmbedding(bitmap, bbox, landmarks)
        saveEmbeddings(listOf(embeddings), "embeddings.json")

        Log.d("MainActivity", "Face embedding: ${embeddings.joinToString()}")

        recognitionOrtSession.close()
        ortEnv.close()
    }

    private fun loadModel(modelName: String): ByteArray {
        assets.open(modelName).use { inputStream ->
            return inputStream.readBytes()
        }
    }

    private fun saveEmbeddings(embeddings: List<FloatArray>, fileName: String) {
        val jsonArray = JSONArray()

        for (embedding in embeddings) {
            val jsonObject = JSONObject()
            val embeddingArray = JSONArray()

            for (value in embedding) {
                embeddingArray.put(value)
            }

            jsonObject.put("embedding", embeddingArray)
            jsonArray.put(jsonObject)
        }

        val jsonObject = JSONObject()
        jsonObject.put("embeddings", jsonArray)

        val jsonFile = File(filesDir, fileName)
        saveJson(jsonObject, jsonFile)

        Log.d("MainActivity", "Embeddings JSON file saved at: ${jsonFile.absolutePath}")
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
