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
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import java.io.File
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections
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
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as BoundingBox

        if (!coordinates.contentEquals(other.coordinates)) return false
        if (confidence != other.confidence) return false
        if (!landmarks.contentEquals(other.landmarks)) return false
        if (classNum != other.classNum) return false
        return true
    }

    override fun hashCode(): Int {
        var result = coordinates.contentHashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + landmarks.contentHashCode()
        return result
    }

    override fun toString(): String {
        return "BoundingBox(coordinates=${coordinates.joinToString(", ")}, confidence=$confidence, landmarks=${landmarks.joinToString(", ")})"
    }
}

internal class ObjectDetection(private val context: Context) {

    fun detect(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result? {
        try {
            Log.d("ObjectDetection", "Starting detection")
            val rawImageBytes = inputStream.readBytes()
            val bitmap = BitmapFactory.decodeByteArray(rawImageBytes, 0, rawImageBytes.size)
            Log.d("ObjectDetection", "Loaded bitmap from input stream")

            val preprocessedImage = preprocessImage(bitmap)
            Log.d("ObjectDetection", "Preprocessed image")

            val expectedSize = 3 * 640 * 640
            val actualSize = preprocessedImage.capacity()
            if (actualSize != expectedSize) {
                Log.e("ObjectDetection", "Preprocessed image size mismatch: expected $expectedSize, but got $actualSize")
                return null
            } else {
                Log.d("ObjectDetection", "Preprocessed image size is correct: $actualSize")
            }

            preprocessedImage.rewind()
            Log.d("ObjectDetection", "FloatBuffer rewound before tensor creation")

            val shape = longArrayOf(1, 3, 640, 640)
            val inputTensor = OnnxTensor.createTensor(ortEnv, preprocessedImage, shape)
            Log.d("ModelInference", "Input tensor shape: ${inputTensor.info.shape.joinToString()}")
            Log.d("ModelInference", "Input tensor values: ${inputTensor.floatBuffer.asFloatArray().joinToString(limit = 10)}")
            Log.d("PrepareInputTensor", "Input tensor data type: ${inputTensor.info.type}")

            Log.d("ObjectDetection", "Running ONNX session with input tensor...")
            val output = ortSession.run(Collections.singletonMap("input", inputTensor), setOf("output"))

            Log.d("ObjectDetection", "ONNX session execution completed")

            output.use {
                val rawOutput = output[0].value
                Log.d("ObjectDetection", "Model output type: ${rawOutput.javaClass.simpleName}")

                if (rawOutput is Array<*>) {
                    val rawOutputArray = rawOutput as Array<Array<FloatArray>>
                    Log.d("ObjectDetection", "Model output: Array<Array<FloatArray>> of size ${rawOutputArray.size}")

                    logRawOutputArrayValues(rawOutputArray)

                    // Process output to get the best bounding box
                    val bestBox = processOutput(rawOutputArray, bitmap.width, bitmap.height)
                    if (bestBox == null) {
                        Log.d("ObjectDetection", "No boxes detected after NMS.")
                        return null
                    }

                    // Draw bounding box on the bitmap
                    val outputBitmap = drawBoundingBox(bitmap, listOf(bestBox))
                    Log.d("ObjectDetection", "Drew bounding box on the bitmap")

                    // Save output image and JSON
                    saveOutputImage(outputBitmap)
                    saveOutputJson(listOf(bestBox))

                    return Result(outputBitmap, listOf(bestBox))
                } else {
                    Log.e("ObjectDetection", "Model output is not of the expected type.")
                    return null
                }
            }
        } catch (e: Exception) {
            Log.e("ObjectDetection", "Error detecting objects", e)
            return null
        }
    }

    private fun logRawOutputArrayValues(rawOutputArray: Array<Array<FloatArray>>) {
        val maxLogEntries = 10 // Limit the number of entries to log to avoid overwhelming the log
        var loggedEntries = 0

        for (i in rawOutputArray.indices) {
            for (j in rawOutputArray[i].indices) {
                val boxData = rawOutputArray[i][j]
                Log.d("ObjectDetection", "Box [$i][$j]: ${boxData.joinToString()}")
                loggedEntries++
                if (loggedEntries >= maxLogEntries) {
                    Log.d("ObjectDetection", "Logged first $maxLogEntries entries.")
                    return
                }
            }
        }
    }

    private fun FloatBuffer.asFloatArray(): FloatArray {
        rewind()
        return FloatArray(remaining()).also { get(it) }
    }

    private fun preprocessImage(bitmap: Bitmap): FloatBuffer {
        Log.d("PreprocessImage", "Starting preprocessImage")

        // Resize the bitmap
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        Log.d("PreprocessImage", "Bitmap resized to 1280x1280")

        // Allocate the float buffer
        val floatBuffer = FloatBuffer.allocate(3 * 640 * 640)
        Log.d("PreprocessImage", "FloatBuffer allocated with capacity: ${floatBuffer.capacity()}")

        // Initialize int array for pixel values
        val intValues = IntArray(640 * 640)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)
        Log.d("PreprocessImage", "Pixel values extracted from resized bitmap")

        var pixel = 0
        for (i in 0 until 640) {
            for (j in 0 until 640) {
                val value = intValues[pixel++]
                val normalizedRed = (value shr 16 and 0xFF) / 255.0f
                floatBuffer.put(normalizedRed)
            }
        }


        pixel = 0
        for (i in 0 until 640) {
            for (j in 0 until 640) {
                val value = intValues[pixel++]
                val normalizedGreen = (value shr 8 and 0xFF) / 255.0f
                floatBuffer.put(normalizedGreen)

            }
        }

        pixel = 0
        for (i in 0 until 640) {
            for (j in 0 until 640) {
                val value = intValues[pixel++]
                val normalizedBlue = (value and 0xFF) / 255.0f
                floatBuffer.put(normalizedBlue)
            }

        }
        Log.d("PreprocessImage", "All pixel values processed and added to FloatBuffer")

        // Validation and logging
        if (floatBuffer.position() != 3 * 640 * 640) {
            Log.e("PreprocessImage", "FloatBuffer size mismatch: expected ${3 * 1280 * 1280}, but got ${floatBuffer.position()}")
        } else {
            Log.d("PreprocessImage", "FloatBuffer size is correct: ${floatBuffer.position()}")
        }

        floatBuffer.rewind()
        Log.d("PreprocessImage", "FloatBuffer rewound to position 0")

        // Log first few values of the float buffer
        val floatArray = FloatArray(10)
        floatBuffer.get(floatArray, 0, 10)
        Log.d("PreprocessImage", "First 10 values of preprocessed image: ${floatArray.joinToString()}")

        Log.d("PreprocessImage", "preprocessImage completed")

        return floatBuffer
    }

    private fun drawBoundingBox(bitmap: Bitmap, boundingBoxes: List<BoundingBox>): Bitmap {
        Log.d("DrawBoundingBox", "Starting drawBoundingBox")

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

        for (box in boundingBoxes) {
            val rect = Rect(box.coordinates[0].toInt(), box.coordinates[1].toInt(), box.coordinates[2].toInt(), box.coordinates[3].toInt())
            canvas.drawRect(rect, paint)
            for (i in box.landmarks.indices step 2) {
                val cx = box.landmarks[i]
                val cy = box.landmarks[i + 1]
                canvas.drawCircle(cx, cy, 6f, landmarkPaint)
            }
        }
        Log.d("DrawBoundingBox", "Finished drawing bounding boxes")

        return outputBitmap
    }

    private fun processOutput(rawOutput: Array<Array<FloatArray>>, originalWidth: Int, originalHeight: Int): BoundingBox? {
        Log.d("ProcessOutput", "Starting processOutput")

        val boxes = ArrayList<BoundingBox>()
        for (output in rawOutput[0]) {
            val confidence = output[4]
            if (confidence > 0.6f) { // Adjust the threshold as necessary
                val left = output[0]
                val top = output[1]
                val right = output[2]
                val bottom = output[3]
                val landmarks = output.sliceArray(5..14)
                val classNum = output[15]

                // Convert to RectF format [left, top, right, bottom]
                val coordinates = floatArrayOf(left, top, right, bottom)

                val box = BoundingBox(coordinates, confidence, landmarks, classNum)
                boxes.add(box)
            }
        }

        Log.d("ProcessOutput", "Initial bounding boxes count: ${boxes.size}")

        val nmsThreshold = 0.5f
        val filteredBoxes = non_max_suppression_face(boxes, 0.6f, nmsThreshold)
        Log.d("ProcessOutput", "Filtered bounding boxes count: ${filteredBoxes.size}")

        if (filteredBoxes.isNotEmpty()) {
            val bestBox = filteredBoxes.maxByOrNull { it.confidence }
            bestBox?.let {
                Log.d("ProcessOutput", "Best box before scaling: $it")
                val gain = min(640.0f / originalWidth, 640.0f / originalHeight)
                val pad = floatArrayOf((640 - originalWidth * gain) / 2, (640 - originalHeight * gain) / 2)
                scale_coords(bestBox.coordinates, originalWidth, originalHeight, gain, pad)
                scale_coords_landmarks(bestBox.landmarks, originalWidth, originalHeight, gain, pad)
                clip_coords(bestBox.coordinates, originalWidth, originalHeight)
                Log.d("ProcessOutput", "Best box after scaling: $bestBox")
            }
            return bestBox
        }
        return null
    }



    private fun non_max_suppression_face(boundingBoxes: List<BoundingBox>, confThreshold: Float, iouThreshold: Float): List<BoundingBox> {
        Log.d("NMS", "Starting non_max_suppression_face with ${boundingBoxes.size} boxes")

        // Step 1: Filter boxes by confidence threshold
        val filteredBoxes = boundingBoxes.filter { it.confidence > confThreshold }
        Log.d("NMS", "After confidence filtering: ${filteredBoxes.size} boxes")

        if (filteredBoxes.isEmpty()) {
            return emptyList()
        }

        // Step 2: Sort boxes by confidence in descending order
        val sortedBoxes = filteredBoxes.sortedByDescending { it.confidence }.toMutableList()

        // Step 3: Prepare the output list
        val selectedBoxes = mutableListOf<BoundingBox>()

        // Logging coordinates before NMS
        Log.d("NMS", "Before NMS: ${sortedBoxes.map { it.coordinates.toList() }}")

        // Step 4: Apply Non-Maximum Suppression (NMS)
        while (sortedBoxes.isNotEmpty()) {
            // Select the box with the highest confidence
            val bestBox = sortedBoxes.removeAt(0)
            selectedBoxes.add(bestBox)

            // Remove boxes that have high IoU with the best box
            val bestBoxRect = floatArrayToRectF(bestBox.coordinates)
            sortedBoxes.removeAll { box -> calculateIoU(bestBoxRect, floatArrayToRectF(box.coordinates)) > iouThreshold }
        }

        // Logging coordinates after NMS
        Log.d("NMS", "After NMS: ${selectedBoxes.take(1).map { it.coordinates.toList() }}")

        Log.d("NMS", "Finished non_max_suppression_face with ${selectedBoxes.size} selected boxes")

        // Step 5: Return only the box with the highest confidence
        return selectedBoxes.take(1)
    }


    private fun floatArrayToRectF(array: FloatArray): RectF {
        return RectF(array[0], array[1], array[2], array[3])
    }

    // Convert RectF to FloatArray
    private fun rectFToFloatArray(rect: RectF): FloatArray {
        return floatArrayOf(rect.left, rect.top, rect.right, rect.bottom)
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

    private fun scale_coords(coords: FloatArray, originalWidth: Int, originalHeight: Int, gain: Float, pad: FloatArray) {
        coords[0] = (coords[0] - pad[0]) / gain
        coords[1] = (coords[1] - pad[1]) / gain
        coords[2] = (coords[2] - pad[0]) / gain
        coords[3] = (coords[3] - pad[1]) / gain
    }

    private fun scale_coords_landmarks(landmarks: FloatArray, originalWidth: Int, originalHeight: Int, gain: Float, pad: FloatArray) {
        for (i in landmarks.indices step 2) {
            landmarks[i] = (landmarks[i] - pad[0]) / gain
            landmarks[i + 1] = (landmarks[i + 1] - pad[1]) / gain
        }
    }

    private fun clip_coords(coords: FloatArray, originalWidth: Int, originalHeight: Int) {
        coords[0] = coords[0].coerceIn(0f, originalWidth.toFloat() - 1)
        coords[1] = coords[1].coerceIn(0f, originalHeight.toFloat() - 1)
        coords[2] = coords[2].coerceIn(0f, originalWidth.toFloat() - 1)
        coords[3] = coords[3].coerceIn(0f, originalHeight.toFloat() - 1)
    }

    private fun saveOutputImage(outputBitmap: Bitmap) {
        val outputDir = File(context.filesDir, "images")
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        val outputFile = File(outputDir, "output_image.jpg")
        outputBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputFile.outputStream())
        Log.d("ObjectDetection", "Saved output image to ${outputFile.absolutePath}")
    }

    private fun saveOutputJson(boundingBoxes: List<BoundingBox>) {
        val outputDir = File(context.filesDir, "json")
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        val outputFile = File(outputDir, "output.json")
        val json = boundingBoxes.joinToString(separator = ",\n", prefix = "[\n", postfix = "\n]") { it.toString() }
        outputFile.writeText(json)
        Log.d("ObjectDetection", "Saved output JSON to ${outputFile.absolutePath}")
    }
}