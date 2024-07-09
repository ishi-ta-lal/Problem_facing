package com.example.detect

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.annotation.SuppressLint
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

            // Rewind the FloatBuffer to ensure it is in the correct state
            preprocessedImage.rewind()
            Log.d("ObjectDetection", "FloatBuffer rewound before tensor creation")

            // Additional logging to verify buffer state
            val remainingElements = preprocessedImage.remaining()
            Log.d("ObjectDetection", "FloatBuffer remaining elements: $remainingElements")

            val shape = longArrayOf(1, 3, 640, 640)
            val inputTensor = OnnxTensor.createTensor(ortEnv, preprocessedImage, shape)
            Log.d("ModelInference", "Input tensor shape: ${inputTensor.info.shape.joinToString()}")
            Log.d("ModelInference", "Input tensor values: ${inputTensor.floatBuffer.asFloatArray().joinToString(limit = 10)}")
            Log.d("PrepareInputTensor", "Input tensor data type: ${inputTensor.info.type}") // Log the data type

            Log.d("ObjectDetection", "Running ONNX session with input tensor...")
            val output = ortSession.run(Collections.singletonMap("input", inputTensor), setOf("output"))

            Log.d("ObjectDetection", "ONNX session execution completed")

            output.use {
                val rawOutput = output[0].value
                Log.d("ObjectDetection", "Model output type: ${rawOutput.javaClass.simpleName}")

                if (rawOutput is Array<*>) {
                    Log.d("ObjectDetection", "Model output is an array with size ${rawOutput.size}")
                    if (rawOutput.isNotEmpty() && rawOutput[0] is Array<*>) {
                        val innerArray = rawOutput[0] as Array<*>
                        Log.d("ObjectDetection", "Inner array size: ${innerArray.size}")
                        if (innerArray.isNotEmpty() && innerArray[0] is FloatArray) {
                            val floatArray = innerArray[0] as FloatArray
                            Log.d("ObjectDetection", "FloatArray size: ${floatArray.size}")
                            Log.d("ObjectDetection", "First FloatArray values: ${floatArray.joinToString()}")
                        } else {
                            Log.e("ObjectDetection", "Inner array does not contain FloatArrays as expected")
                        }
                    } else {
                        Log.e("ObjectDetection", "Model output does not contain the expected inner arrays")
                    }

                    @Suppress("UNCHECKED_CAST")
                    val rawOutputArray = rawOutput as Array<Array<FloatArray>>
                    Log.d("ObjectDetection", "Model output: Array<Array<FloatArray>> of size ${rawOutputArray.size}")

                    logRawOutputArrayValues(rawOutputArray)

                    val boxes = processOutput(rawOutputArray)
                    Log.d("ObjectDetection", "Processed ${boxes.size} bounding boxes")

                    if (boxes.isEmpty()) {
                        Log.d("ObjectDetection", "No boxes detected.")
                    }

                    val outputBitmap = drawBoundingBox(bitmap, boxes)
                    Log.d("ObjectDetection", "Drew bounding boxes on the bitmap")

                    saveOutputImage(outputBitmap)
                    saveOutputJson(boxes)

                    return Result(outputBitmap, boxes)
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
                val normalizedGreen = (value shr 8 and 0xFF) / 255.0f
                val normalizedBlue = (value and 0xFF) / 255.0f

                floatBuffer.put(normalizedRed)
                floatBuffer.put(normalizedGreen)
                floatBuffer.put(normalizedBlue)
            }
            if (i % 100 == 0) {
                Log.d("PreprocessImage", "Processed row: $i")
            }
        }
        Log.d("PreprocessImage", "All pixel values processed and added to FloatBuffer")

        // Validation and logging
        if (floatBuffer.position() != 3 * 640 * 640) {
            Log.e("PreprocessImage", "FloatBuffer size mismatch: expected ${3 * 640 * 640}, but got ${floatBuffer.position()}")
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




    private fun FloatBuffer.asFloatArray(): FloatArray {
        rewind()
        return FloatArray(remaining()).also { get(it) }
    }

    private fun processOutput(rawOutput: Array<Array<FloatArray>>): List<BoundingBox> {
        val boundingBoxes = mutableListOf<BoundingBox>()
        var highestConfidenceBox: BoundingBox? = null

        Log.d("ObjectDetection", "Processing model output with size: ${rawOutput[0].size}")

        Log.d("ObjectDetection", "Starting to process model output with size: ${rawOutput[0].size}")

        for (boxIndex in rawOutput[0].indices) {
            val boxData = rawOutput[0][boxIndex]

            val coordinates = floatArrayOf(boxData[0], boxData[1], boxData[2], boxData[3])
            val confidence = boxData[4]
            val classNum = boxData[15]


            if (confidence > 0.00015f) {  // Adjusted confidence threshold
                val landmarks = floatArrayOf(
                    boxData[5], boxData[6], boxData[7], boxData[8],
                    boxData[9], boxData[10], boxData[11], boxData[12],
                    boxData[13], boxData[14]
                )
                val boundingBox = BoundingBox(coordinates, confidence, landmarks, classNum)
                boundingBoxes.add(boundingBox)

                Log.d("ObjectDetection", "Box $boxIndex - Landmarks: ${landmarks.joinToString()}")
                Log.d("ObjectDetection", "Box $boxIndex - Added to bounding boxes")

                // Keep track of the highest confidence box
                if (highestConfidenceBox == null || confidence > highestConfidenceBox.confidence) {
                    highestConfidenceBox = boundingBox
                    Log.d("ObjectDetection", "Box $boxIndex - Updated highest confidence box")
                }
            }
//            else {
//                Log.d("ObjectDetection", "Box $boxIndex - Did not pass confidence threshold")
//            }
        }

        Log.d("ObjectDetection", "Total bounding boxes detected: ${boundingBoxes.size}")
        val result = highestConfidenceBox?.let { listOf(it) } ?: emptyList()
        Log.d("ObjectDetection", "Returning ${result.size} bounding box(es)")

        return result
    }




    @SuppressLint("DefaultLocale")
    fun drawBoundingBox(bitmap: Bitmap, boundingBoxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
//        val paint = Paint()
        val textPaint = Paint()
        val pointPaint = Paint().apply { color = Color.RED }
        val squarePaint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.GREEN
            strokeWidth = 4f
        }

        textPaint.color = Color.WHITE
        textPaint.textSize = 30f
        textPaint.style = Paint.Style.FILL

        for (box in boundingBoxes) {
            val (x1, y1, x2, y2) = box.coordinates.map { it.toInt() }

            // Draw confidence text near the first coordinate point
            val confidenceText = String.format("%.3f", box.confidence)
            canvas.drawText(confidenceText, x1.toFloat(), y1 - 10f, textPaint)

            // Draw points at the coordinates
            canvas.drawCircle(x1.toFloat(), y1.toFloat(), 5f, pointPaint)
            canvas.drawCircle(x2.toFloat(), y2.toFloat(), 5f, pointPaint)

            // Draw points at the landmarks
            for (i in box.landmarks.indices step 2) {
                val pointX = box.landmarks[i].toInt()
                val pointY = box.landmarks[i + 1].toInt()
                canvas.drawCircle(pointX.toFloat(), pointY.toFloat(), 5f, pointPaint)
            }

            // Draw a square around the face
            canvas.drawRect(Rect(x1, y1, x2, y2), squarePaint)
        }

        return mutableBitmap
    }



    private fun saveOutputImage(outputBitmap: Bitmap) {
        val outputImageFile = File(context.getExternalFilesDir(null), "output_image.png")
        outputImageFile.outputStream().use {
            outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        Log.d("ObjectDetection", "Saved output image to ${outputImageFile.absolutePath}")
    }

    private fun saveOutputJson(boxes: List<BoundingBox>) {
        val outputJsonFile = File(context.getExternalFilesDir(null), "output_boxes.json")
        outputJsonFile.outputStream().use {
            it.write(boxes.toString().toByteArray())
        }
        Log.d("ObjectDetection", "Saved bounding boxes to ${outputJsonFile.absolutePath}")
    }
}