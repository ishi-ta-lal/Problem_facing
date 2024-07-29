package com.example.frs

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class Recognition(private val ortSession: OrtSession, private val ortEnv: OrtEnvironment) {

    fun preprocess(image: Bitmap, bbox: List<Float>?, landmarks: List<Float>?, kwargs: Map<String, Any> = mapOf()): Mat {
        Log.d("Recognition", "Starting preprocessing")

        // Convert Bitmap to Mat
        val mat = Mat()
        Utils.bitmapToMat(image, mat)
        Log.d("Recognition", "Initial image shape: ${mat.size()}")
        Log.d("Recognition", "Initial image values: ${mat.dump()}")
        Log.d("Recognition", "Initial image type: ${CvType.typeToString(mat.type())}")

        var M: Mat? = null
        var imageSize = mutableListOf<Int>()
        val strImageSize = kwargs["image_size"]?.toString() ?: "112,112"
        if (strImageSize.isNotEmpty()) {
            imageSize = strImageSize.split(',').map { it.toInt() }.toMutableList()
            if (imageSize.size == 1) {
                imageSize = mutableListOf(imageSize[0], imageSize[0])
            }
            require(imageSize.size == 2) { "Image size must have exactly 2 elements" }
        }
        Log.d("Recognition", "Image size: $imageSize")

        if (landmarks != null) {
            require(imageSize.size == 2) { "Image size must be defined for landmark transformation" }

            val landmarkArray = Array(landmarks.size / 2) { DoubleArray(2) }
            for (i in landmarks.indices step 2) {
                landmarkArray[i / 2] = doubleArrayOf(landmarks[i].toDouble(), landmarks[i + 1].toDouble())
            }
            Log.d("Recognition", "Landmarks array: ${landmarkArray.contentDeepToString()}")

            M = transform(mat, landmarkArray, strImageSize)
            Log.d("Recognition", "Transformation matrix M: ${M.dump()}")
        }

        return if (M == null) {
            val det = bbox?.let {
                intArrayOf(it[0].toInt(), it[1].toInt(), it[2].toInt(), it[3].toInt())
            } ?: run {
                intArrayOf(
                    (mat.cols() * 0.0625).toInt(),
                    (mat.rows() * 0.0625).toInt(),
                    mat.cols() - (mat.cols() * 0.0625).toInt(),
                    mat.rows() - (mat.rows() * 0.0625).toInt()
                )
            }
            Log.d("Recognition", "Bounding box: ${det.contentToString()}")

            val margin = kwargs["margin"]?.toString()?.toInt() ?: 44
            val bb = intArrayOf(
                max(det[0] - margin / 2, 0),
                max(det[1] - margin / 2, 0),
                min(det[2] + margin / 2, mat.cols()),
                min(det[3] + margin / 2, mat.rows())
            )
            Log.d("Recognition", "Bounding box with margin: ${bb.contentToString()}")

            val ret = Mat(mat, Rect(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]))
            Log.d("Recognition", "Cropped image shape: ${ret.size()}")
            Log.d("Recognition", "Cropped image values: ${ret.dump()}")
            Log.d("Recognition", "Cropped image type: ${CvType.typeToString(ret.type())}")
            if (imageSize.isNotEmpty()) {
                Imgproc.resize(ret, ret, Size(imageSize[1].toDouble(), imageSize[0].toDouble()))
                Log.d("Recognition", "Resized image shape: ${ret.size()}")
                Log.d("Recognition", "Resized image values: ${ret.dump()}")
                Log.d("Recognition", "Resized image type: ${CvType.typeToString(ret.type())}")
            }
            ret
        } else {
            require(imageSize.size == 2) { "Image size must be defined for warping" }
            val warped = Mat()
            val borderValue = Scalar(0.0)  // Set border value to 0.0 (black)
            Imgproc.warpAffine(mat, warped, M, Size(imageSize[1].toDouble(), imageSize[0].toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, borderValue)
            Log.d("Recognition", "Warped image shape: ${warped.size()}")
            Log.d("Recognition", "Warped image values: ${warped.dump()}")
            Log.d("Recognition", "Warped image type: ${CvType.typeToString(warped.type())}")
            warped
        }
    }

    private fun transform(img: Mat, landmark: Array<DoubleArray>, imageSize: String): Mat {
        val srcPoints = arrayOf(
            Point(30.2946, 51.6963),
            Point(65.5318, 51.5014),
            Point(48.0252, 71.7366),
            Point(33.5493, 92.3655),
            Point(62.7299, 92.2041)
        )

        if (imageSize == "112,112") {
            srcPoints.forEach { it.x += 8.0 }
        }
        Log.d("Recognition", "Source points: ${srcPoints.contentToString()}")

        val src = MatOfPoint2f(*srcPoints)
        val dstPoints = landmark.map { Point(it[0], it[1]) }.toTypedArray()
        val dst = MatOfPoint2f(*dstPoints)
        Log.d("Recognition", "Destination points: ${dstPoints.contentToString()}")

        val tform = Calib3d.estimateAffinePartial2D(dst, src)
        return tform
    }

    fun getEmbedding(image: Bitmap, bbox: List<Float>, landmarks: List<Float>): FloatArray {
        Log.d("Recognition", "Starting getEmbedding")

        val kwargs = mapOf("image_size" to "112,112")
        val preprocessedMat = preprocess(image, bbox, landmarks, kwargs)
        Log.d("Recognition", "Preprocessed image size: ${preprocessedMat.size()}")
        Log.d("Recognition", "Preprocessed image values: ${preprocessedMat.dump()}")
        Log.d("Recognition", "Preprocessed image type: ${CvType.typeToString(preprocessedMat.type())}")

        Imgproc.cvtColor(preprocessedMat, preprocessedMat, Imgproc.COLOR_BGR2RGB)
        Log.d("Recognition", "Converted to RGB image size: ${preprocessedMat.size()}")
        Log.d("Recognition", "Converted to RGB image values: ${preprocessedMat.dump()}")
        Log.d("Recognition", "Converted to RGB image type: ${CvType.typeToString(preprocessedMat.type())}")

        val alignedMat = Mat()
        Core.transpose(preprocessedMat, alignedMat)
        Core.flip(alignedMat, alignedMat, 1)
        alignedMat.convertTo(alignedMat, CvType.CV_32F)
        Log.d("Recognition", "Aligned image size: ${alignedMat.size()}")
        Log.d("Recognition", "Aligned image values: ${alignedMat.dump()}")
        Log.d("Recognition", "Aligned image type: ${CvType.typeToString(alignedMat.type())}")

        val inputBlob = FloatArray(3 * alignedMat.rows() * alignedMat.cols())
        alignedMat.get(0, 0, inputBlob)
        Log.d("Recognition", "Input blob values: ${inputBlob.contentToString()}")

        val tensorShape = longArrayOf(1, 3, alignedMat.rows().toLong(), alignedMat.cols().toLong())
        val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputBlob), tensorShape)
        Log.d("Recognition", "Created input blob shape: [1, 3, ${alignedMat.rows()}, ${alignedMat.cols()}]")
        Log.d("Recognition", "Created input blob dtype: ${tensor.info.type}")

        val results = ortSession.run(mapOf(ortSession.inputNames.first() to tensor))
        val embedding = (results[0] as OnnxTensor).floatBuffer.array()
        Log.d("Recognition", "ONNX model inference result shape: ${embedding.size}")
        Log.d("Recognition", "ONNX model inference result values: ${embedding.contentToString()}")

        val norm = sqrt(embedding.fold(0.0f) { acc, value -> acc + value * value })
        val normalizedEmbedding = embedding.map { it / norm }.toFloatArray()
        Log.d("Recognition", "Normalized embedding shape: ${normalizedEmbedding.size}")
        Log.d("Recognition", "Normalized embedding values: ${normalizedEmbedding.contentToString()}")

        Log.d("Recognition", "Embedding calculation completed")
        return normalizedEmbedding
    }
}
