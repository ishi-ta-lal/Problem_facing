package com.example.detect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import java.io.InputStream
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

internal data class Result(
    var outputBitmap: Bitmap,
    var outputBoxes: List<BoundingBox>
)

internal data class BoundingBox(
    val coordinates: FloatArray,
    val confidence: Float,
    val landmarks: FloatArray,
    val classNum: Float
) {
    override fun toString(): String {
        return "BoundingBox(coordinates=${coordinates.joinToString(", ")}, confidence=$confidence, landmarks=${landmarks.joinToString(", ")})"
    }
}

internal class ObjectDetection(private val context: Context) {

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result? {
        val startTime = System.nanoTime()
        val startMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
        return try {
            val rawImageBytes = inputStream.readBytes()
            val bitmap = BitmapFactory.decodeByteArray(rawImageBytes, 0, rawImageBytes.size)

            val preprocessedImage = preprocessImage(bitmap)
            val expectedSize = 3 * 640 * 640
            if (preprocessedImage.capacity() != expectedSize) {
                Log.e("ObjectDetection", "Preprocessed image size mismatch")
                return null
            }

            preprocessedImage.rewind()
            val shape = longArrayOf(1, 3, 640, 640)
            val inputTensor = OnnxTensor.createTensor(ortEnv, preprocessedImage, shape)

            val output = ortSession.run(mapOf("input" to inputTensor), setOf("output"))
            output.use {
                val rawOutput = output[0].value
                if (rawOutput is Array<*>) {
                    val rawOutputArray = rawOutput as Array<Array<FloatArray>>
                    val bestBox = processOutput(rawOutputArray, bitmap.width, bitmap.height) ?: return null

                    val outputBitmap = drawBoundingBox(bitmap, listOf(bestBox))
                    saveOutput(outputBitmap, listOf(bestBox))

                    val endTime = System.nanoTime()
                    val endMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
                    val duration = (endTime - startTime) / 1_000_000.0 // convert to milliseconds
                    val memoryUsage = (endMemory - startMemory) / 1024.0 // convert to KB

                    Log.d("ObjectDetection", "Time taken: $duration ms")
                    Log.d("ObjectDetection", "Memory used: $memoryUsage KB")

                    return Result(outputBitmap, listOf(bestBox))
                } else {
                    Log.e("ObjectDetection", "Model output is not of the expected type.")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e("ObjectDetection", "Error detecting objects", e)
            null
        }
    }

    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val floatBuffer = FloatBuffer.allocate(3 * 640 * 640)

        val intValues = IntArray(640 * 640)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        intValues.forEach { value ->
            floatBuffer.put((value shr 16 and 0xFF) / 255.0f)
        }
        intValues.forEach { value ->
            floatBuffer.put((value shr 8 and 0xFF) / 255.0f)
        }
        intValues.forEach { value ->
            floatBuffer.put((value and 0xFF) / 255.0f)
        }

        floatBuffer.rewind()
        return floatBuffer
    }

    private fun drawBoundingBox(bitmap: Bitmap, boundingBoxes: List<BoundingBox>): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 3f
        }
        val landmarkPaint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.BLUE
            strokeWidth = 6f
        }

        boundingBoxes.forEach { box ->
            val rect = RectF(box.coordinates[0], box.coordinates[1], box.coordinates[2], box.coordinates[3])
            canvas.drawRect(rect, paint)

            // Draw landmarks
            for (i in box.landmarks.indices step 2) {
                val cx = box.landmarks[i]
                val cy = box.landmarks[i + 1]
                canvas.drawCircle(cx, cy, 6f, landmarkPaint)
            }
        }

        return outputBitmap
    }


    private fun processOutput(rawOutput: Array<Array<FloatArray>>, originalWidth: Int, originalHeight: Int): BoundingBox? {
        val boxes = mutableListOf<BoundingBox>()
        rawOutput[0].forEach { output ->
            if (output[4] > 0.6f) {
                val (centerX, centerY, width, height) = output.slice(0..3)
                val left = centerX - width / 2
                val top = centerY - height / 2
                val right = centerX + width / 2
                val bottom = centerY + height / 2

                val box = BoundingBox(
                    coordinates = floatArrayOf(left, top, right, bottom),
                    confidence = output[4],
                    landmarks = output.sliceArray(5..14),
                    classNum = output[15]
                )
                boxes.add(box)
            }
        }

        val filteredBoxes = nonMaxSuppressionFace(boxes, 0.6f, 0.5f)
        val bestBox = filteredBoxes.maxByOrNull { it.confidence } ?: return null

        val gain = min(640.0f / originalWidth, 640.0f / originalHeight)
        val pad = floatArrayOf((640 - originalWidth * gain) / 2, (640 - originalHeight * gain) / 2)
        scaleCoords(bestBox.coordinates, gain, pad)
        scaleCoordsLandmarks(bestBox.landmarks, gain, pad)
        clipCoords(bestBox.coordinates, originalWidth, originalHeight)

        return bestBox
    }

    private fun nonMaxSuppressionFace(
        predictions: List<BoundingBox>,
        confThreshold: Float,
        iouThreshold: Float
    ): List<BoundingBox> {
        val filteredBoxes = predictions.filter { it.confidence > confThreshold }
        if (filteredBoxes.isEmpty()) return emptyList()

        val selectedBoxes = mutableListOf<BoundingBox>()
        val boxes = filteredBoxes.map { RectF(it.coordinates[0], it.coordinates[1], it.coordinates[2], it.coordinates[3]) }
        val scores = filteredBoxes.map { it.confidence }

        val indices = nms(boxes, scores, iouThreshold)
        indices.maxByOrNull { filteredBoxes[it].confidence }?.let { selectedBoxes.add(filteredBoxes[it]) }

        return selectedBoxes
    }

    private fun nms(boxes: List<RectF>, scores: List<Float>, iouThreshold: Float): List<Int> {
        val sortedIndices = scores.indices.sortedByDescending { scores[it] }.toMutableList()
        val selectedIndices = mutableListOf<Int>()

        while (sortedIndices.isNotEmpty()) {
            val currentIndex = sortedIndices.removeAt(0)
            selectedIndices.add(currentIndex)
            val currentBox = boxes[currentIndex]

            sortedIndices.removeAll { index ->
                calculateIoU(currentBox, boxes[index]) > iouThreshold
            }
        }

        return selectedIndices
    }

    private fun calculateIoU(boxA: RectF, boxB: RectF): Float {
        val intersection = RectF(
            max(boxA.left, boxB.left),
            max(boxA.top, boxB.top),
            min(boxA.right, boxB.right),
            min(boxA.bottom, boxB.bottom)
        )

        val intersectionArea = if (intersection.left < intersection.right && intersection.top < intersection.bottom) {
            (intersection.right - intersection.left) * (intersection.bottom - intersection.top)
        } else {
            0f
        }

        val areaA = (boxA.right - boxA.left) * (boxA.bottom - boxA.top)
        val areaB = (boxB.right - boxB.left) * (boxB.bottom - boxB.top)
        val unionArea = areaA + areaB - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    private fun scaleCoords(coords: FloatArray, gain: Float, pad: FloatArray) {
        coords[0] = (coords[0] - pad[0]) / gain
        coords[1] = (coords[1] - pad[1]) / gain
        coords[2] = (coords[2] - pad[0]) / gain
        coords[3] = (coords[3] - pad[1]) / gain
    }

    private fun scaleCoordsLandmarks(
        landmarks: FloatArray,
        gain: Float,
        pad: FloatArray
    ) {
        for (i in landmarks.indices step 2) {
            landmarks[i] = (landmarks[i] - pad[0]) / gain
            landmarks[i + 1] = (landmarks[i + 1] - pad[1]) / gain
        }
    }

    private fun clipCoords(coords: FloatArray, width: Int, height: Int) {
        coords[0] = max(0f, min(coords[0], width.toFloat() - 1))
        coords[1] = max(0f, min(coords[1], height.toFloat() - 1))
        coords[2] = max(0f, min(coords[2], width.toFloat() - 1))
        coords[3] = max(0f, min(coords[3], height.toFloat() - 1))
    }

    private fun saveOutput(outputBitmap: Bitmap, boundingBoxes: List<BoundingBox>) {
        Log.d("ObjectDetection", "Detected bounding boxes: ${boundingBoxes.joinToString("\n")}")
        val outputPath = "${context.filesDir}/detected_image.jpg"
        val outputStream = context.openFileOutput("detected_image.jpg", Context.MODE_PRIVATE)
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
        outputStream.close()
        Log.d("ObjectDetection", "Detected image saved to $outputPath")
    }
}