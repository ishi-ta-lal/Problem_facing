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
import android.util.Log
import java.io.File
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections
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
        Log.d("PreprocessImage", "Bitmap resized to 640x640")

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
            Log.e("PreprocessImage", "FloatBuffer size mismatch: expected ${3 * 640 * 640}, but got ${floatBuffer.position()}")
        } else {
            Log.d("PreprocessImage", "FloatBuffer size is correct: ${floatBuffer.position()}")
        }

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
            strokeWidth = 3f
        }

        for (box in boundingBoxes) {
            val rect = Rect(
                box.coordinates[0].toInt(),
                box.coordinates[1].toInt(),
                box.coordinates[2].toInt(),
                box.coordinates[3].toInt()
            )
            canvas.drawRect(rect, paint)

            // Draw landmarks
            for (i in box.landmarks.indices step 2) {
                val cx = box.landmarks[i]
                val cy = box.landmarks[i + 1]
                canvas.drawCircle(cx, cy, 3f, landmarkPaint)
            }
        }
        Log.d("DrawBoundingBox", "Drew ${boundingBoxes.size} bounding boxes and landmarks on the bitmap")

        return outputBitmap
    }

    private fun saveOutputImage(bitmap: Bitmap) {
        val outputDir = File(context.getExternalFilesDir(null), "output")
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        val outputFile = File(outputDir, "output_image.png")
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputFile.outputStream())
        Log.d("SaveOutputImage", "Output image saved to: ${outputFile.absolutePath}")
    }

    private fun saveOutputJson(boxes: List<BoundingBox>) {
        val outputDir = File(context.getExternalFilesDir(null), "output")
        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }
        val outputFile = File(outputDir, "output_boxes.json")
        outputFile.bufferedWriter().use { out ->
            out.write("[\n")
            boxes.forEachIndexed { index, box ->
                val boxString = """
                    {
                        "coordinates": [${box.coordinates.joinToString()}],
                        "confidence": ${box.confidence},
                        "landmarks": [${box.landmarks.joinToString()}],
                        "classNum": ${box.classNum}
                    }
                """.trimIndent()
                out.write(boxString)
                if (index != boxes.size - 1) {
                    out.write(",\n")
                }
            }
            out.write("\n]")
        }
        Log.d("SaveOutputJson", "Output boxes saved to: ${outputFile.absolutePath}")
    }

    private fun processOutput(outputArray: Array<Array<FloatArray>>, imageWidth: Int, imageHeight: Int): BoundingBox? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        // Iterate through each output array
        for (batchArray in outputArray) {
            // Iterate through each box data in the batch
            for (boxData in batchArray) {
                val coordinates = boxData.sliceArray(0..3)
                val confidence = boxData[4]
                val landmarks = boxData.sliceArray(5..14)
                val classNum = boxData[15]

                if (confidence > 0.6) {
                    val boundingBox = BoundingBox(coordinates, confidence, landmarks, classNum)
                    boundingBoxes.add(boundingBox)
                }
            }
        }

        // Apply NMS
        val nmsBoxes = nms(boundingBoxes, 0.6f, 0.5f)

        // Sort by confidence and take the box with highest confidence
        val sortedBoxes = nmsBoxes.sortedByDescending { it.confidence }
        val bestBox = sortedBoxes.firstOrNull()

        // Scale coordinates of the best box
        bestBox?.let {
            val scaledBox = scaleCoordsLandmarks(it, imageWidth, imageHeight)
            return scaledBox
        }

        return null
    }

    private fun scaleCoordsLandmarks(bbox: BoundingBox, width: Int, height: Int): BoundingBox {
        Log.d("ScaleCoordsLandmarks", "Original coordinates: ${bbox.coordinates.joinToString()}")
        Log.d("ScaleCoordsLandmarks", "Original landmarks: ${bbox.landmarks.joinToString()}")

        val gain = min(640.0f / width, 640.0f / height)
        val padX = (640 - width * gain) / 2
        val padY = (640 - height * gain) / 2

        val scaledCoordinates = bbox.coordinates.mapIndexed { index, coord ->
            val scaledValue = when (index % 2) {
                0 -> (coord - padX) / gain
                else -> (coord - padY) / gain
            }
            "%.1f".format(scaledValue).toFloat()  // Rounding to one decimal place
        }.toFloatArray()

        val scaledLandmarks = bbox.landmarks.mapIndexed { index, landmark ->
            val scaledValue = when (index % 2) {
                0 -> (landmark - padX) / gain
                else -> (landmark - padY) / gain
            }
            "%.1f".format(scaledValue).toFloat()  // Rounding to one decimal place
        }.toFloatArray()

        val roundedConfidence = "%.1f".format(bbox.confidence).toFloat()  // Rounding to one decimal place
        val roundedClassNum = "%.1f".format(bbox.classNum).toFloat()  // Rounding to one decimal place

        val scaledBoundingBox = BoundingBox(scaledCoordinates, roundedConfidence, scaledLandmarks, roundedClassNum)

        Log.d("ScaleCoordsLandmarks", "Scaled coordinates: ${scaledBoundingBox.coordinates.joinToString()}")
        Log.d("ScaleCoordsLandmarks", "Scaled landmarks: ${scaledBoundingBox.landmarks.joinToString()}")

        return scaledBoundingBox
    }

    private fun nms(boxes: List<BoundingBox>, confThres: Float, iouThres: Float): List<BoundingBox> {
        // Convert to the necessary format and filter by confidence

        val filteredBoxes = boxes.filter { it.confidence > confThres }
            .sortedByDescending { it.confidence }

        val selectedBoxes = mutableListOf<BoundingBox>()
        val suppressed = BooleanArray(filteredBoxes.size)

        for (i in filteredBoxes.indices) {
            if (suppressed[i]) continue
            val boxA = filteredBoxes[i]
            selectedBoxes.add(boxA)

            for (j in i + 1 until filteredBoxes.size) {
                if (suppressed[j]) continue
                val boxB = filteredBoxes[j]

                // Calculate IoU
                val iou = boxIou(xywh2xyxy(boxA.coordinates), xywh2xyxy(boxB.coordinates))
                if (iou > iouThres) {
                    suppressed[j] = true
                }
            }
        }

        // Logging for debugging
        Log.d("NMS", "After NMS:")
        selectedBoxes.forEach { box ->
            Log.d("NMS", "Coordinates: ${box.coordinates.joinToString()}, Confidence: ${box.confidence}")
        }

        return selectedBoxes
    }


    private fun xywh2xyxy(box: FloatArray): FloatArray {
        val x = box[0]
        val y = box[1]
        val w = box[2]
        val h = box[3]
        return floatArrayOf(
            x - w / 2, // x1
            y - h / 2, // y1
            x + w / 2, // x2
            y + h / 2  // y2
        )
    }

    private fun boxArea(box: FloatArray): Float {
        val width = box[2] - box[0]
        val height = box[3] - box[1]
        return width * height
    }

    private fun boxIou(box1: FloatArray, box2: FloatArray): Float {
        val inter = floatArrayOf(
            maxOf(box1[0], box2[0]), // x1
            maxOf(box1[1], box2[1]), // y1
            minOf(box1[2], box2[2]), // x2
            minOf(box1[3], box2[3])  // y2
        )

        val interArea = maxOf(inter[2] - inter[0], 0f) * maxOf(inter[3] - inter[1], 0f)
        val box1Area = boxArea(box1)
        val box2Area = boxArea(box2)

        return interArea / (box1Area + box2Area - interArea)
    }

}