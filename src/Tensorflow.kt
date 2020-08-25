package org.sourcekey.NorikoAI.Calligrapher

import kotlin.js.Json
import org.khronos.webgl.*
import org.w3c.dom.*
import org.w3c.files.File
import kotlin.js.Promise


open external class Rank

open external class R0: Rank

open external class R1: Rank

open external class R2: Rank

open external class R3: Rank

open external class R4: Rank

open external class R5: Rank

open external class R6: Rank

open external class Tensor{
    fun flatten(): Tensor1D

    fun asScalar(): Scalar

    fun as1D(): Tensor1D

    fun as2D(rows: Number, columns: Number): Tensor2D

    fun as3D(rows: Number, columns: Number, depth: Number): Tensor3D

    fun as4D(rows: Number, columns: Number, depth: Number, depth2: Number): Tensor4D

    fun as5D(rows: Number, columns: Number, depth: Number, depth2: Number, depth3: Number): Tensor5D

    fun asType(dtype: String): Tensor

    fun buffer(): Promise<TensorBuffer>

    fun bufferSync(): TensorBuffer

    fun array(): Promise<DoubleArray>

    fun arraySync(): Array<Number>

    fun data(): Promise<Array<dynamic>>

    fun dataSync(): Array<dynamic>

    fun dispose()

    fun toFloat(): Tensor

    fun toInt(): Tensor

    fun toBool(): Tensor

    fun reshapeAs(x: Tensor): Tensor

    fun squeeze(): Tensor

    fun toString(verbose: Boolean = definedExternally): String

    //Tensors / Creation

    fun clone(): Tensor

    fun onesLike(): Tensor

    fun print(verbose: Boolean = definedExternally)

    fun zerosLike(): Tensor

    //Tensors / Transformations

    fun batchToSpaceND(blockShape: Array<Int>, crops: Array<DoubleArray>): Tensor

    fun broadcastTo(shape: Array<Int>): Tensor

    fun cast(dtype: String): Tensor

    fun depthToSpace(blockSize: Int, dataFormat: String = definedExternally): Tensor4D

    fun expandDims(axis: Number): Tensor

    fun pad(paddings: Array<DoubleArray>, constantValue: Number = definedExternally): Tensor

    fun reshape(shape: Array<Int>): Tensor

    fun setdiff1dAsync(y: Tensor): Promise<Array<Tensor>>

    fun spaceToBatchND(blockShape: Array<Int>, paddings: Array<DoubleArray>): Tensor

    fun squeeze(axis: Array<Number>): Tensor

    //Tensors / Slicing and Joining

    fun gather(indices: Tensor, axis: Number = definedExternally): Tensor

    fun reverse(axis: Number): Tensor

    fun reverse(axis: Array<Number>): Tensor

    fun slice(begin: Number, size: Number): Tensor

    fun slice(begin: Number, size: Array<Number>): Tensor

    fun slice(begin: Array<Number>, size: Number): Tensor

    fun slice(begin: Array<Number>, size: Array<Number>): Tensor

    fun split(numOrSizeSplits: Array<Int>, axis: Number = definedExternally): Array<Tensor>

    fun split(numOrSizeSplits: Int, axis: Number = definedExternally): Array<Tensor>

    fun tile(reps: Array<Number>): Tensor

    fun unstack(axis: Number = definedExternally): Array<Tensor>

    //Operations / Arithmetic

    fun add(b: Tensor): Tensor

    fun sub(b: Tensor): Tensor

    fun mul(b: Tensor): Tensor

    fun div(b: Tensor): Tensor

    fun divNoNan(b: Tensor): Tensor

    fun floorDiv(b: Tensor): Tensor

    fun maximum(b: Tensor): Tensor

    fun minimum(b: Tensor): Tensor

    fun mod(b: Tensor): Tensor

    fun pow(exp: Tensor): Tensor

    fun squaredDifference(b: Tensor): Tensor

    //Operations / Basic math

    fun abs(): Tensor

    fun acos(): Tensor

    fun acosh(): Tensor

    fun asin(): Tensor

    fun asinh(): Tensor

    fun atan(): Tensor

    fun atan2(b: Tensor): Tensor

    fun atanh(): Tensor

    fun ceil(): Tensor

    fun clipByValue(clipValueMin: Number, clipValueMax: Number): Tensor

    fun cos(): Tensor

    fun cosh(): Tensor

    fun elu(): Tensor

    fun erf(): Tensor

    fun exp(): Tensor

    fun expm1(): Tensor

    fun floor(): Tensor

    fun isFinite(): Tensor

    fun isInf(): Tensor

    fun isNaN(): Tensor

    fun leakyRelu(alpha: Number = definedExternally): Tensor

    fun log(): Tensor

    fun log1p(): Tensor

    fun logSigmoid(): Tensor

    fun neg(): Tensor

    fun prelu(alpha: Number): Tensor

    fun reciprocal(): Tensor

    fun relu(): Tensor

    fun relu6(): Tensor

    fun round(): Tensor

    fun rsqrt(): Tensor

    fun selu(): Tensor

    fun sigmoid(): Tensor

    fun sign(): Tensor

    fun sin(): Tensor

    fun sinh(): Tensor

    fun softplus(): Tensor

    fun sqrt(): Tensor

    fun square(): Tensor

    fun step(alpha: Number = definedExternally): Tensor

    fun tan(): Tensor

    fun tanh(): Tensor

    //Operations / Matrices

    fun dot(t2: Tensor): Tensor

    fun matMul(b: Tensor, transposeA: Boolean = definedExternally, transposeB: Boolean = definedExternally): Tensor

    fun norm(): Tensor

    fun norm(ord: Number = definedExternally, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun norm(ord: Number = definedExternally, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun norm(ord: String = definedExternally, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun norm(ord: String = definedExternally, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun transpose(perm: Array<Number> = definedExternally): Tensor

    //Operations / Reduction

    fun all(): Tensor

    fun all(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun all(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun any(): Tensor

    fun any(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun any(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun argMax(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun argMin(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun logSumExp(): Tensor

    fun logSumExp(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun logSumExp(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun max(): Tensor

    fun max(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun max(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun mean(): Tensor

    fun mean(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun mean(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun min(): Tensor

    fun min(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun min(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun prod(): Tensor

    fun prod(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun prod(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun sum(): Tensor

    fun sum(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun sum(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    //Operations / Normalization

    fun batchNorm(mean: Tensor, variance: Tensor): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor1D): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor1D): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(mean: Tensor1D, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun logSoftmax(axis: Number = definedExternally): Tensor

    fun moments(): Json

    fun moments(axis: Number = definedExternally, keepDims: Boolean = definedExternally): Json

    fun moments(axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Json

    fun softmax(dim: Number = definedExternally): Tensor

    //Operations / Logical

    fun equal(b: Tensor): Tensor?

    fun greater(b: Tensor): Tensor?

    fun greaterEqual(b: Tensor): Tensor?

    fun less(b: Tensor): Tensor?

    fun lessEqual(b: Tensor): Tensor?

    fun logicalAnd(b: Tensor): Tensor

    fun logicalNot(): Tensor

    fun logicalOr(b: Tensor): Tensor

    fun logicalXor(b: Tensor): Tensor

    fun notEqual(b: Tensor): Tensor?

    fun where(condition: Tensor, b: Tensor): Tensor?

    //Operations / Scan

    fun cumsum(axis: Number = definedExternally, exclusive: Boolean = definedExternally, reverse: Boolean = definedExternally): Tensor

    //Operations / Evaluation

    fun topk(k: Number = definedExternally, sorted: Boolean = definedExternally): Json?

    //Operations / Segment

    fun unsortedSegmentSum(segmentIds: Tensor1D, numSegments: Number): Tensor

    //Operations / Moving Average

    fun movingAverage(v: Tensor, decay: Number): Tensor?

    fun movingAverage(v: Tensor, decay: Number, step: Number = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    fun movingAverage(v: Tensor, decay: Number, step: Scalar = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    fun movingAverage(v: Tensor, decay: Scalar): Tensor?

    fun movingAverage(v: Tensor, decay: Scalar, step: Number = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    fun movingAverage(v: Tensor, decay: Scalar, step: Scalar = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    //Operations / Slicing and Joining

    fun gatherND(indices: Tensor): Tensor?

    fun stridedSlice(begin: Array<Number>, end: Array<Number>, strides: Array<Number> = definedExternally, beginMask: Number = definedExternally, endMask: Number = definedExternally, ellipsisMask: Number = definedExternally, newAxisMask: Number = definedExternally, shrinkAxisMask: Number = definedExternally): Tensor?

    //Operations / Dropout

    fun dropout(rate: Number): Tensor

    fun dropout(rate: Number, noiseShape: Array<Number> = definedExternally, seed: Number = definedExternally): Tensor

    fun dropout(rate: Number, noiseShape: Array<Number> = definedExternally, seed: String = definedExternally): Tensor
}

open external class Scalar: Tensor

open external class Tensor1D: Tensor

open external class Tensor2D: Tensor

open external class Tensor3D: Tensor

open external class Tensor4D: Tensor

open external class Tensor5D: Tensor

open external class Tensor6D: Tensor

open external class TensorBuffer{
    fun set(value: Array<dynamic>, vararg locs: Int): TensorBuffer

    fun get(vararg locs: Int): Array<dynamic>

    fun toTensor(): Tensor
}

//typealias Tensor = Tensor<Rank>

//typealias Scalar = Tensor<R0>

//typealias Tensor1D = Tensor<R1>

//typealias Tensor2D = Tensor<R2>

//typealias Tensor3D = Tensor<R3>

//typealias Tensor4D = Tensor<R4>

//typealias Tensor5D = Tensor<R5>

//typealias Tensor6D = Tensor<R6>

open external class Serializable

open external class Serialization

open external class ConfigDict

open external class SymbolicTensor

open external class Layer: Serializable{
    fun apply(inputs: Tensor, kwargs: Json = definedExternally): Tensor

    fun apply(inputs: Array<Tensor>, kwargs: Json = definedExternally): Array<Tensor>

    fun apply(inputs: SymbolicTensor, kwargs: Json = definedExternally): SymbolicTensor

    fun apply(inputs: Array<SymbolicTensor>, kwargs: Json = definedExternally): Array<SymbolicTensor>

    fun countParams(): Number

    fun build(inputShape: Array<Int?>)

    fun build(inputShape: Array<Array<Int?>>)

    fun getWeights(trainableOnly: Boolean = definedExternally): Array<Tensor>

    fun setWeights(weights: Array<Tensor>)

    fun addWeight(name: String, shape: Array<Int?>, dtype: String = definedExternally, initializer: Initializer = definedExternally, regularizer: Regularizer = definedExternally, trainable: Boolean = definedExternally, constraint: Constraint = definedExternally): LayerVariable

    fun addLoss(losses: RegularizerFn)

    fun addLoss(losses: Array<RegularizerFn>)

    fun computeOutputShape(inputShape: Array<Int?>): Array<Int?>

    fun computeOutputShape(inputShape: Array<Array<Int?>>): Array<Array<Int?>>

    fun getConfig(): ConfigDict

    fun dispose(): DisposeResult
}

open external class RNNCell: Layer

open external class Bidirectional

open external class AlphaDropout

open external class GaussianDropout

open external class GaussianNoise

open external class Layers{

    //Layers / Advanced Activation

    fun elu(args: Json = definedExternally): Layer

    fun leakyReLU(args: Json = definedExternally): Layer

    fun prelu(args: Json = definedExternally): Layer

    fun reLU(args: Json = definedExternally): Layer

    fun softmax(args: Json = definedExternally): Layer

    fun thresholdedReLU(args: Json = definedExternally): Layer

    //Layers / Basic

    fun activation(args: Json): Layer

    fun dense(args: Json): Layer

    fun dropout(args: Json): Layer

    fun embedding(args: Json): Layer

    fun flatten(args: Json = definedExternally): Layer

    fun permute(args: Json): Layer

    fun repeatVector(args: Json): Layer

    fun reshape(args: Json): Layer

    fun spatialDropout1d(args: Json): Layer

    //Layers / Convolutional

    fun conv1d(args: Json): Layer

    fun conv2d(args: Json): Layer

    fun conv2dTranspose(args: Json): Layer

    fun conv3d(args: Json): Layer

    fun cropping2D(args: Json): Layer

    fun depthwiseConv2d(args: Json): Layer

    fun separableConv2d(args: Json): Layer

    fun upSampling2d(args: Json): Layer

    //Layers / Merge

    fun add(args: Json = definedExternally): Layer

    fun average(args: Json = definedExternally): Layer

    fun concatenate(args: Json = definedExternally): Layer

    fun dot(args: Json = definedExternally): Layer

    fun maximum(args: Json = definedExternally): Layer

    fun minimum(args: Json = definedExternally): Layer

    fun multiply(args: Json = definedExternally): Layer

    //Layers / Normalization

    fun batchNormalization(args: Json = definedExternally): Layer

    fun layerNormalization(args: Json = definedExternally): Layer

    //Layers / Pooling

    fun averagePooling1d(args: Json): Layer

    fun averagePooling2d(args: Json): Layer

    fun averagePooling3d(args: Json): Layer

    fun globalAveragePooling1d(args: Json = definedExternally): Layer

    fun globalAveragePooling2d(args: Json): Layer

    fun globalMaxPooling1d(args: Json = definedExternally): Layer

    fun globalMaxPooling2d(args: Json): Layer

    fun maxPooling1d(args: Json): Layer

    fun maxPooling2d(args: Json): Layer

    fun maxPooling3d(args: Json): Layer

    //Layers / Recurrent

    fun gru(args: Json): Layer

    fun gruCell(args: Json): RNNCell

    fun lstm(args: Json): Layer

    fun lstmCell(args: Json): RNNCell

    fun rnn(args: Json): Layer

    fun simpleRNN(args: Json): Layer

    fun simpleRNNCell(args: Json): RNNCell

    fun stackedRNNCells(args: Json): RNNCell

    //Layers / Wrapper

    fun bidirectional(args: Json): Bidirectional

    fun timeDistributed(args: Json): Layer

    //Layers / Inputs

    fun inputLayer(args: Json): Layer

    //Layers / Padding

    fun zeroPadding2d(args: Json = definedExternally): Layer

    //Layers / Noise

    fun alphaDropout(args: Json): AlphaDropout

    fun gaussianDropout(args: Json): GaussianDropout

    fun gaussianNoise(args: Json): GaussianNoise

    //Layers / Mask

    fun masking(args: Json = definedExternally): Layer
}

open external class Initializer

open external class Zeros

open external class Initializers: Serializable{
    fun constant(args: Json): Initializer

    fun glorotNormal(args: Json): Initializer

    fun glorotUniform(args: Json): Initializer

    fun heNormal(args: Json): Initializer

    fun heUniform(args: Json): Initializer

    fun identity(args: Json): Initializer

    fun leCunNormal(args: Json): Initializer

    fun leCunUniform(args: Json): Initializer

    fun ones(): Initializer

    fun orthogonal(args: Json): Initializer

    fun randomNormal(args: Json): Initializer

    fun randomUniform(args: Json): Initializer

    fun truncatedNormal(args: Json): Initializer

    fun varianceScaling(args: Json): Initializer

    fun zeros(): Zeros
}

open external class Regularizer

open external class Constraint: Serializable

open external class Constraints{
    fun maxNorm(args: Json): Constraint

    fun minMaxNorm(config: Json): Constraint

    fun nonNeg(): Constraint

    fun unitNorm(args: Json): Constraint
}

open external class LayerVariable

open external class RegularizerFn

open external class DisposeResult

open external class Dataset{
    fun batch(batchSize: Number, smallLastBatch: Boolean = definedExternally): Dataset

    fun concatenate(dataset: Dataset): Dataset

    fun filter(predicate:(value: dynamic)->Boolean): Dataset //predicate((value: T) => boolean) https://js.tensorflow.org/api/latest/#tf.data.Dataset.filter

    fun forEachAsync(f:(input: dynamic)->Unit): Promise<Unit> //f((input: T) => void) https://js.tensorflow.org/api/latest/#tf.data.Dataset.forEachAsync

    fun map(transform:(value: dynamic)->Unit): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.map

    fun map(transform:(value: dynamic)->Number): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.map

    fun map(transform:(value: dynamic)->String): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.map

    fun map(transform:(value: dynamic)->Tensor): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.map

    fun map(transform:(value: dynamic)->Array<Tensor>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.map

    fun map(transform:(value: dynamic)->Json): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.map

    fun mapAsync(transform:(value: dynamic)->Promise<Unit>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.mapAsync

    fun mapAsync(transform:(value: dynamic)->Promise<Number>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.mapAsync

    fun mapAsync(transform:(value: dynamic)->Promise<String>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.mapAsync

    fun mapAsync(transform:(value: dynamic)->Promise<Tensor>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.mapAsync

    fun mapAsync(transform:(value: dynamic)->Promise<Array<Tensor>>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.mapAsync

    fun mapAsync(transform:(value: dynamic)->Promise<Json>): Dataset //value: T https://js.tensorflow.org/api/latest/#tf.data.Dataset.mapAsync

    fun prefetch(bufferSize: Number): Dataset

    fun repeat(count: Number = definedExternally): Dataset

    fun skip(count: Number): Dataset

    fun shuffle(bufferSize: Number, seed: String = definedExternally, reshuffleEachIteration: Boolean = definedExternally): Dataset

    fun take(count: Number): Dataset

    fun toArray(): Promise<Array<dynamic>> //Returns: Promise<T[]>
}

open external class RequestInfo

open external class CSVDataset: Dataset{
    fun columnNames(): Promise<Array<String>>
}

open external class MicrophoneIterator

open external class WebcamIterator

open external class DatasetContainer

open external class Data{
    //Data / Creation

    fun array(items: Unit): Dataset

    fun array(items: Number): Dataset

    fun array(items: String): Dataset

    fun array(items: Tensor): Dataset

    fun array(items: Array<Tensor>): Dataset

    fun array(items: Json): Dataset

    fun csv(source: RequestInfo, csvConfig: Json = definedExternally): CSVDataset

    fun generator(generator:()->Iterator<dynamic>): Dataset

    fun generator(generator:()->Promise<Iterator<Unit>>): Dataset

    fun generator(generator:()->Promise<Iterator<Number>>): Dataset

    fun generator(generator:()->Promise<Iterator<String>>): Dataset

    fun generator(generator:()->Promise<Iterator<Tensor>>): Dataset

    fun generator(generator:()->Promise<Iterator<Array<Tensor>>>): Dataset

    fun generator(generator:()->Promise<Iterator<Json>>): Dataset

    fun microphone(microphoneConfig: Json = definedExternally): Promise<MicrophoneIterator>

    fun webcam(webcamVideoElement: HTMLVideoElement = definedExternally, webcamConfig: Json = definedExternally): Promise<WebcamIterator>

    //Data / Operations

    fun zip(datasets: DatasetContainer): Dataset

    //Aata / Classes


}

open external class IOHandler

open external class ModelArtifactsInfo

open external class ModelArtifacts

open external class SaveResult

open external class IO{

    //Models / Loading

    fun browserDownloads(fileNamePrefix: String = definedExternally): IOHandler

    fun browserFiles(files: Array<File>): IOHandler

    fun http(path: String, loadOptions: Json = definedExternally): IOHandler

    //Models / Management

    fun copyModel(sourceURL: String, destURL: String): Promise<ModelArtifactsInfo>

    fun listModels(): Promise<dynamic>

    fun moveModel(sourceURL: String, destURL: String): Promise<ModelArtifactsInfo>

    fun removeModel(url: String): Promise<ModelArtifactsInfo>
}

open external class Variable: Tensor{
    fun assign(newValue: Tensor)
}

open external class LayersModel{
    //inline printFn:(message: dynamic, optionalParams: dynamic)->Unit)

    fun compile(args: Json)

    fun evaluate(x: Tensor, y: Tensor, args: Json = definedExternally): Scalar

    fun evaluate(x: Array<Tensor>, y: Array<Tensor>, args: Json = definedExternally): Array<Scalar>

    //fun evaluateDataset(dataset: Data.Dataset, args: Json = definedExternally): Promise<Scalar|Array<Scalar>>

    fun predict(x: Tensor, args: Json = definedExternally): Tensor

    fun predict(x: Array<Tensor>, args: Json = definedExternally): Array<Tensor>

    fun predictOnBatch(x: Tensor): Tensor

    fun predictOnBatch(x: Array<Tensor>): Array<Tensor>

    fun fit(x: Tensor, y: Tensor, args: Json = definedExternally): Promise<History>

    fun fit(x: Tensor, y: Array<Tensor>, args: Json = definedExternally): Promise<History>

    fun fit(x: Tensor, y: Json, args: Json = definedExternally): Promise<History>

    fun fit(x: Array<Tensor>, y: Tensor, args: Json = definedExternally): Promise<History>

    fun fit(x: Array<Tensor>, y: Array<Tensor>, args: Json = definedExternally): Promise<History>

    fun fit(x: Array<Tensor>, y: Json, args: Json = definedExternally): Promise<History>

    fun fit(x: Json, y: Tensor, args: Json = definedExternally): Promise<History>

    fun fit(x: Json, y: Array<Tensor>, args: Json = definedExternally): Promise<History>

    fun fit(x: Json, y: Json, args: Json = definedExternally): Promise<History>

    fun fitDataset(dataset: Dataset, args: Json): Promise<History>

    fun trainOnBatch(x: Tensor, y: Tensor): Promise<Double>

    fun trainOnBatch(x: Array<Tensor>, y: Array<Tensor>): Promise<DoubleArray>

    fun save(handlerOrURL: IOHandler, config: Json = definedExternally): Promise<SaveResult>

    fun save(handlerOrURL: String, config: Json = definedExternally): Promise<SaveResult>

    fun getLayer(): Layer

    fun getLayer(name: String): Layer

    fun getLayer(name: String, index: Int): Layer
}

open external class Sequential: LayersModel{
    fun add(layer: Layer)
}

open external class InferenceModel

open external class GraphModel: InferenceModel{
    fun loadSync(artifacts: ModelArtifacts): Boolean

    fun save(handlerOrURL: IOHandler, config: Json = definedExternally): Promise<SaveResult>

    fun save(handlerOrURL: String, config: Json = definedExternally): Promise<SaveResult>

    fun predict(inputs: Tensor, config: Json): Tensor

    fun predict(inputs: Array<Tensor>, config: Json): Array<Tensor>

    fun predict(inputs: Json, config: Json): Json

    fun execute(inputs: Tensor, outputs: String): Tensor

    fun execute(inputs: Tensor, outputs: Array<String>): Tensor

    fun execute(inputs: Array<Tensor>, outputs: String): Array<Tensor>

    fun execute(inputs: Array<Tensor>, outputs: Array<String>): Array<Tensor>

    fun executeAsync(inputs: Tensor, outputs: String): Promise<Tensor>

    fun executeAsync(inputs: Tensor, outputs: Array<String>): Promise<Tensor>

    fun executeAsync(inputs: Array<Tensor>, outputs: String): Promise<Array<Tensor>>

    fun executeAsync(inputs: Array<Tensor>, outputs: Array<String>): Promise<Array<Tensor>>

    fun dispose()
}

open external class SerializableConstructor

open external class OpMapper

open external class Image{
    fun cropAndResize(image: Tensor4D, boxes: Tensor2D, boxInd: Tensor1D, cropSize: Array<Int>, method: String = definedExternally, extrapolationValue: Number = definedExternally): Tensor4D

    fun nonMaxSuppressionAsync(boxes: Tensor2D, scores: Tensor1D, maxOutputSize: Number, iouThreshold: Number = definedExternally, scoreThreshold: Number = definedExternally): Promise<Tensor1D>

    fun nonMaxSuppressionWithScore(boxes: Tensor2D, scores: Tensor1D, maxOutputSize: Number, iouThreshold: Number = definedExternally, scoreThreshold: Number = definedExternally, softNmsSigma: Number = definedExternally): Json

    fun resizeBilinear(images: Tensor3D, size: Array<Int>, alignCorners: Boolean = definedExternally): Tensor3D

    fun resizeBilinear(images: Tensor4D, size: Array<Int>, alignCorners: Boolean = definedExternally): Tensor4D

    fun resizeNearestNeighbor(images: Tensor3D, size: Array<Int>, alignCorners: Boolean = definedExternally): Tensor3D

    fun resizeNearestNeighbor(images: Tensor4D, size: Array<Int>, alignCorners: Boolean = definedExternally): Tensor4D
}

open external class Spectral{
    fun fft(input: Tensor): Tensor?

    fun ifft(input: Tensor): Tensor?

    fun irfft(input: Tensor): Tensor?

    fun rfft(input: Tensor, fftLength: Number = definedExternally): Tensor?
}

open external class Signal{
    fun frame(signal: Tensor1D, frameLength: Number, frameStep: Number, padEnd: Boolean = definedExternally, padValue: Number = definedExternally): Tensor

    fun hammingWindow(windowLength: Int): Tensor1D

    fun hannWindow(windowLength: Int): Tensor1D

    fun stft(signal: Tensor1D, frameLength: Int, frameStep: Int, fftLength: Int = definedExternally, windowFn:(length: Number)->Tensor1D = definedExternally): Tensor
}

open external class Linalg{
    fun bandPart(a: Tensor, numLower: Int, numUpper: Int): Tensor

    fun gramSchmidt(xs: Array<Tensor1D>): Array<Tensor1D>

    fun gramSchmidt(xs: Array<Tensor2D>): Array<Tensor2D>

    fun qr(x: Tensor, fullMatrices: Boolean = definedExternally): Array<Tensor>?
}

open external class SGDOptimizer

open external class MomentumOptimizer

open external class AdagradOptimizer

open external class AdadeltaOptimizer

open external class AdamOptimizer

open external class AdamaxOptimizer

open external class RMSPropOptimizer

open external class Optimizer: Serializable{
    fun minimize(f:()->Scalar, returnCost: Boolean = definedExternally, varList: Array<Variable> = definedExternally): Scalar?
}

open external class Train{
    fun sgd(learningRate: Number): SGDOptimizer

    fun momentum(learningRate: Number, momentum: Number, useNesterov: Boolean = definedExternally): MomentumOptimizer

    fun adagrad(learningRate: Number, initialAccumulatorValue: Number = definedExternally): AdagradOptimizer

    fun adadelta(learningRate: Number = definedExternally, rho: Number = definedExternally, epsilon: Number = definedExternally): AdadeltaOptimizer

    fun adam(learningRate: Number = definedExternally, beta1: Number = definedExternally, beta2: Number = definedExternally, epsilon: Number = definedExternally): AdamOptimizer

    fun adamax(learningRate: Number = definedExternally, beta1: Number = definedExternally, beta2: Number = definedExternally, epsilon: Number = definedExternally, decay: Number = definedExternally): AdamaxOptimizer

    fun rmsprop(learningRate: Number, decay: Number = definedExternally, momentum: Number = definedExternally, epsilon: Number = definedExternally, centered: Boolean = definedExternally): RMSPropOptimizer
}

open external class Reduction

open external class Losses{
    fun absoluteDifference(labels: Tensor, predictions: Tensor, weights: Tensor = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun computeWeightedLoss(losses: Tensor, weights: Tensor = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun cosineDistance(labels: Tensor, predictions: Tensor, axis: Number, weights: Tensor = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun hingeLoss(labels: Tensor, predictions: Tensor, weights: Tensor = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun huberLoss(labels: Tensor, predictions: Tensor, weights: Tensor = definedExternally, delta: Number = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun logLoss(labels: Tensor, predictions: Tensor, weights: Tensor = definedExternally, epsilon: Number = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun meanSquaredError(labels: Tensor, predictions: Tensor, weights: Tensor = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun sigmoidCrossEntropy(multiClassLabels: Tensor, logits: Tensor, weights: Tensor = definedExternally, labelSmoothing: Number = definedExternally, reduction: Reduction = definedExternally): Tensor

    fun softmaxCrossEntropy(onehotLabels: Tensor, logits: Tensor, weights: Tensor = definedExternally, labelSmoothing: Number = definedExternally, reduction: Reduction = definedExternally): Tensor
}

open external class MemoryInfo{
    val numBytes: Number

    val numTensors: Number

    val numDataBuffers: Number

    val unreliable: Boolean

    val reasons: Array<String>

    val numBytesInGPU: Number
}

open external class TimingInfo{
    val wallMs: Number //Wall execution time.

    val kernelMs: Number //Kernel execution time, ignoring data transfer. If using the WebGL backend and the query timer extension is not available, this will return an error object.

    val uploadWaitMs: Number //CPU blocking time on texture uploads.

    val downloadWaitMs: Number //CPU blocking time on texture downloads(readPixels).
}

open external class ProfileInfo{
    val newBytes: Number //the number of new bytes allocated

    val newTensors: Number //the number of new tensors created

    val peakBytes: Number //the peak number of bytes allocated

    val kernels: Array<dynamic> //an array of objects for each kernel involved that reports their input and output shapes, number of bytes used, and number of new tensors created.
}

open external class Environment

open external class Engine

open external class Regularizers{
    fun l1(config: Json = definedExternally): Regularizer

    fun l1l2(config: Json = definedExternally): Regularizer

    fun l2(config: Json = definedExternally): Regularizer
}

open external class RequestInit

open external class Response

open external class Util{
    fun assert(expr: Boolean, msg:()->String)

    fun createShuffledIndices(n: Number): Uint32Array

    fun decodeString(bytes: Uint8Array, encoding: String = definedExternally): String

    fun encodeString(s: String, encoding: String = definedExternally): Uint8Array

    fun fetch(path: String, requestInits: RequestInit = definedExternally): Promise<Response>

    fun flatten(arr: dynamic, result: dynamic = definedExternally, skipTypedArray: dynamic = definedExternally): dynamic //https://js.tensorflow.org/api/latest/#util.flatten

    fun now(): Number

    fun shuffle(array: Uint32Array)

    fun shuffle(array: Int32Array)

    fun shuffle(array: Float32Array)

    fun shuffle(array: Any)

    fun sizeFromShape(shape: Array<Int>)
}

open external class PixelData

open external class Browser{
    fun fromPixels(pixels: PixelData, numChannels: Number = definedExternally): Tensor3D

    fun fromPixels(pixels: ImageData, numChannels: Number = definedExternally): Tensor3D

    fun fromPixels(pixels: HTMLImageElement, numChannels: Number = definedExternally): Tensor3D

    fun fromPixels(pixels: HTMLCanvasElement, numChannels: Number = definedExternally): Tensor3D

    fun fromPixels(pixels: HTMLVideoElement, numChannels: Number = definedExternally): Tensor3D

    fun toPixels(img: Tensor2D, canvas: HTMLCanvasElement = definedExternally): Promise<Uint8ClampedArray>

    fun toPixels(img: Tensor3D, canvas: HTMLCanvasElement = definedExternally): Promise<Uint8ClampedArray>
}

open external class KernelBackend

open external class Metrics{
    fun binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor

    fun binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor

    fun categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor

    fun categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor

    fun cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor

    fun meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor

    fun meanAbsolutePercentageError(yTrue: Tensor, yPred: Tensor): Tensor

    fun meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor

    fun precision(yTrue: Tensor, yPred: Tensor): Tensor

    fun recall(yTrue: Tensor, yPred: Tensor): Tensor

    fun sparseCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor
}

open external class EarlyStopping

open external class Callbacks{
    fun earlyStopping(args: Json = definedExternally): EarlyStopping
}

@JsModule("@tensorflow/tfjs")
external object Tensorflow{

    //Tensors / Creation

    fun tensor(values: Array<*>, shape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor?

    fun scalar(value: Number, dtype: String = definedExternally): Scalar

    fun scalar(value: String, dtype: String = definedExternally): Scalar

    fun scalar(value: Boolean, dtype: String = definedExternally): Scalar

    fun scalar(value: Array<*>, dtype: String = definedExternally): Scalar?

    fun tensor1d(values: Array<*>, dtype: String = definedExternally): Tensor1D?

    fun tensor2d(values: Array<*>, shape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor2D?

    fun tensor3d(values: Array<*>, shape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor3D?

    fun tensor4d(values: Array<*>, shape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor4D?

    fun tensor5d(values: Array<*>, shape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor5D?

    fun tensor6d(values: Array<*>, shape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor6D?

    fun buffer(shape: Array<Int>, dtype: String = definedExternally, values: List<String> = definedExternally): TensorBuffer

    fun clone(x: Tensor): Tensor

    fun complex(real: Tensor, imag: Tensor): Tensor

    fun eye(numRows: Number, numColumns: Number = definedExternally, batchShape: Array<Int> = definedExternally, dtype: String = definedExternally): Tensor2D

    fun fill(shape: Array<Int>, value: String, dtype: String = definedExternally): Tensor

    fun fill(shape: Array<Int>, value: Int, dtype: String = definedExternally): Tensor

    fun imag(input: Tensor): Tensor

    fun linspace(start: Int, stop: Int, num: Int): Tensor1D

    fun oneHot(indices: Tensor, depth: Number, onValue: Int = definedExternally, offValue: Int = definedExternally): Tensor

    fun ones(shape: Array<Int>, dtype: String = definedExternally): Tensor

    fun onesLike(x: Tensor): Tensor

    fun print(x: Tensor, verbose: Boolean = definedExternally)

    fun range(start: Int, stop: Int, step: Int = definedExternally, dtype: String = definedExternally): Tensor1D

    fun real(input: Tensor): Tensor

    fun truncatedNormal(shape: Array<Int>, mean: Number = definedExternally, stdDev: Number = definedExternally, dtype: String = definedExternally, seed: Number = definedExternally): Tensor

    fun variable(initialValue: Tensor, trainable: Boolean = definedExternally, name: String = definedExternally, dtype: String = definedExternally): Variable

    fun zeros(shape: Array<Int>, dtype: String = definedExternally): Tensor

    fun zerosLike(x: Tensor): Tensor

    //Tensors / Transformations

    fun batchToSpaceND(x: Tensor, blockShape: Array<Int>, crops: Array<DoubleArray>): Tensor

    fun broadcastTo(x: Tensor, shape: Array<Int>): Tensor

    fun cast(x: Tensor, dtype: String): Tensor

    fun depthToSpace(x: Tensor, blockSize: Int, dataFormat: String = definedExternally): Tensor4D

    fun expandDims(x: Tensor, axis: Number): Tensor

    fun pad(x: Tensor, paddings: Array<DoubleArray>, constantValue: Number = definedExternally): Tensor

    fun reshape(x: Tensor, shape: Array<Int>): Tensor

    fun setdiff1dAsync(x: Tensor, y: Tensor): Promise<Array<Tensor>>

    fun spaceToBatchND(x: Tensor, blockShape: Array<Int>, paddings: Array<DoubleArray>): Tensor

    fun squeeze(x: Tensor, axis: Array<Number>): Tensor

    //Tensors / Slicing and Joining

    fun booleanMaskAsync(tensor: Tensor, mask: Tensor, axis: Number = definedExternally): Promise<Tensor>

    fun concat(tensors: Array<Tensor>, axis: Number = definedExternally): Tensor?

    fun gather(x: Tensor, indices: Tensor, axis: Number = definedExternally): Tensor

    fun reverse(x: Tensor, axis: Number): Tensor

    fun reverse(x: Tensor, axis: Array<Number>): Tensor

    fun slice(x: Tensor, begin: Number, size: Number): Tensor

    fun slice(x: Tensor, begin: Number, size: Array<Number>): Tensor

    fun slice(x: Tensor, begin: Array<Number>, size: Number): Tensor

    fun slice(x: Tensor, begin: Array<Number>, size: Array<Number>): Tensor

    fun split(x: Tensor, numOrSizeSplits: Array<Int>, axis: Number = definedExternally): Array<Tensor>

    fun split(x: Tensor, numOrSizeSplits: Int, axis: Number = definedExternally): Array<Tensor>

    fun stack(tensors: Array<Tensor>, axis: Number = definedExternally): Tensor?

    fun tile(x: Tensor, reps: Array<Number>): Tensor

    fun unstack(x: Tensor, axis: Number = definedExternally): Array<Tensor>

    //Tensors / Random

    fun multinomial(logits: Tensor1D, numSamples: Number, seed: Number = definedExternally, normalized: Boolean = definedExternally): Tensor1D

    fun multinomial(logits: Tensor2D, numSamples: Number, seed: Number = definedExternally, normalized: Boolean = definedExternally): Tensor2D

    fun randomGamma(shape: Array<Int>, alpha: Number, beta: Number = definedExternally, dtype: String = definedExternally, seed: Number = definedExternally): Tensor

    fun randomNormal(shape: Array<Int>, mean: Number = definedExternally, stdDev: Number = definedExternally, dtype: String = definedExternally, seed: Number = definedExternally): Tensor

    fun randomUniform(shape: Array<Int>, minval: Number = definedExternally, maxval: Number = definedExternally, dtype: String = definedExternally, seed: Number = definedExternally): Tensor

    fun randomUniform(shape: Array<Int>, minval: Number = definedExternally, maxval: Number = definedExternally, dtype: String = definedExternally, seed: String = definedExternally): Tensor

    //Serialization

    val serialization: Serialization

    //Layers

    val layers: Layers

    //Models / Creation

    fun sequential(config: Json = definedExternally): Sequential

    fun model(args: Json): LayersModel

    //Models / Inputs

    fun input(config: Json): SymbolicTensor

    //Models / Loading

    fun loadGraphModel(modelUrl: String, options: Json = definedExternally): Promise<GraphModel>

    fun loadGraphModel(modelUrl: IOHandler, options: Json = definedExternally): Promise<GraphModel>

    fun loadLayersModel(pathOrIOHandler: String, options: Json = definedExternally): Promise<LayersModel>

    fun loadLayersModel(pathOrIOHandler: IOHandler, options: Json = definedExternally): Promise<LayersModel>

    //Models / Serialization

    fun registerClass(cls: SerializableConstructor)

    //Models / Op Registry

    fun deregisterOp(name: String)

    fun getRegisteredOp(name: String): OpMapper

    fun registerOp(name: String, opFunc: Json)

    //Operations / Arithmetic

    fun add(a: Tensor, b: Tensor): Tensor

    fun sub(a: Tensor, b: Tensor): Tensor

    fun mul(a: Tensor, b: Tensor): Tensor

    fun div(a: Tensor, b: Tensor): Tensor

    fun addN(tensors: Array<Tensor>): Tensor?

    fun divNoNan(a: Tensor, b: Tensor): Tensor

    fun floorDiv(a: Tensor, b: Tensor): Tensor

    fun maximum(a: Tensor, b: Tensor): Tensor

    fun minimum(a: Tensor, b: Tensor): Tensor

    fun mod(a: Tensor, b: Tensor): Tensor

    fun pow(base: Tensor, exp: Tensor): Tensor

    fun squaredDifference(a: Tensor, b: Tensor): Tensor

    //Operations / Basic math

    fun abs(x: Tensor): Tensor

    fun acos(x: Tensor): Tensor

    fun acosh(x: Tensor): Tensor

    fun asin(x: Tensor): Tensor

    fun asinh(x: Tensor): Tensor

    fun atan(x: Tensor): Tensor

    fun atan2(a: Tensor, b: Tensor): Tensor

    fun atanh(x: Tensor): Tensor

    fun ceil(x: Tensor): Tensor

    fun clipByValue(x: Tensor, clipValueMin: Number, clipValueMax: Number): Tensor

    fun cos(x: Tensor): Tensor

    fun cosh(x: Tensor): Tensor

    fun elu(x: Tensor): Tensor

    fun erf(x: Tensor): Tensor

    fun exp(x: Tensor): Tensor

    fun expm1(x: Tensor): Tensor

    fun floor(x: Tensor): Tensor

    fun isFinite(x: Tensor): Tensor

    fun isInf(x: Tensor): Tensor

    fun isNaN(x: Tensor): Tensor

    fun leakyRelu(x: Tensor, alpha: Number = definedExternally): Tensor

    fun log(x: Tensor): Tensor

    fun log1p(x: Tensor): Tensor

    fun logSigmoid(x: Tensor): Tensor

    fun neg(x: Tensor): Tensor

    fun prelu(x: Tensor, alpha: Number): Tensor

    fun reciprocal(x: Tensor): Tensor

    fun relu(x: Tensor): Tensor

    fun relu6(x: Tensor): Tensor

    fun round(x: Tensor): Tensor

    fun rsqrt(x: Tensor): Tensor

    fun selu(x: Tensor): Tensor

    fun sigmoid(x: Tensor): Tensor

    fun sign(x: Tensor): Tensor

    fun sin(x: Tensor): Tensor

    fun sinh(x: Tensor): Tensor

    fun softplus(x: Tensor): Tensor

    fun sqrt(x: Tensor): Tensor

    fun square(x: Tensor): Tensor

    fun step(x: Tensor, alpha: Number = definedExternally): Tensor

    fun tan(x: Tensor): Tensor

    fun tanh(x: Tensor): Tensor

    //Operations / Matrices

    fun dot(t1: Tensor, t2: Tensor): Tensor

    fun matMul(a: Tensor, b: Tensor, transposeA: Boolean = definedExternally, transposeB: Boolean = definedExternally): Tensor

    fun norm(x: Tensor): Tensor

    fun norm(x: Tensor, ord: Number = definedExternally, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun norm(x: Tensor, ord: Number = definedExternally, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun norm(x: Tensor, ord: String = definedExternally, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun norm(x: Tensor, ord: String = definedExternally, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun outerProduct(v1: Tensor1D, v2: Tensor1D): Tensor2D

    fun transpose(x: Tensor, perm: Array<Number> = definedExternally): Tensor

    //Operations / Convolution

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: String): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: Number): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: String): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: Number): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: String): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: Number): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: String): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: Number): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: String): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: Number): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun avgPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun conv1d(x: Tensor2D, filter: Tensor3D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor2D

    fun conv1d(x: Tensor2D, filter: Tensor3D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor2D

    fun conv1d(x: Tensor2D, filter: Tensor3D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor2D

    fun conv1d(x: Tensor3D, filter: Tensor3D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv1d(x: Tensor3D, filter: Tensor3D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv1d(x: Tensor3D, filter: Tensor3D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2d(x: Tensor3D, filter: Tensor4D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2d(x: Tensor3D, filter: Tensor4D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2d(x: Tensor3D, filter: Tensor4D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2d(x: Tensor4D, filter: Tensor4D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv2d(x: Tensor4D, filter: Tensor4D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv2d(x: Tensor4D, filter: Tensor4D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv2dTranspose(x: Tensor3D, filter: Tensor4D, outputShape: Array<Int>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2dTranspose(x: Tensor3D, filter: Tensor4D, outputShape: Array<Int>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2dTranspose(x: Tensor3D, filter: Tensor4D, outputShape: Array<Int>, strides: Number, pad: String, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2dTranspose(x: Tensor3D, filter: Tensor4D, outputShape: Array<Int>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally): Tensor3D

    fun conv2dTranspose(x: Tensor4D, filter: Tensor4D, outputShape: Array<Int>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv2dTranspose(x: Tensor4D, filter: Tensor4D, outputShape: Array<Int>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv2dTranspose(x: Tensor4D, filter: Tensor4D, outputShape: Array<Int>, strides: Number, pad: String, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv2dTranspose(x: Tensor4D, filter: Tensor4D, outputShape: Array<Int>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally): Tensor4D

    fun conv3d(x: Tensor4D, filter: Tensor5D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally): Tensor4D

    fun conv3d(x: Tensor4D, filter: Tensor5D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally): Tensor4D

    fun conv3d(x: Tensor4D, filter: Tensor5D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally): Tensor4D

    fun conv3d(x: Tensor5D, filter: Tensor5D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally): Tensor5D

    fun conv3d(x: Tensor5D, filter: Tensor5D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally): Tensor5D

    fun conv3d(x: Tensor5D, filter: Tensor5D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally): Tensor5D

    fun conv3dTranspose(x: Tensor4D, filter: Tensor5D, outputShape: Array<Int>, strides: Array<Number>, pad: String): Tensor4D

    fun conv3dTranspose(x: Tensor4D, filter: Tensor5D, outputShape: Array<Int>, strides: Array<Number>, pad: Number): Tensor4D

    fun conv3dTranspose(x: Tensor4D, filter: Tensor5D, outputShape: Array<Int>, strides: Number, pad: String): Tensor4D

    fun conv3dTranspose(x: Tensor4D, filter: Tensor5D, outputShape: Array<Int>, strides: Number, pad: Number): Tensor4D

    fun conv3dTranspose(x: Tensor5D, filter: Tensor5D, outputShape: Array<Int>, strides: Array<Number>, pad: String): Tensor5D

    fun conv3dTranspose(x: Tensor5D, filter: Tensor5D, outputShape: Array<Int>, strides: Array<Number>, pad: Number): Tensor5D

    fun conv3dTranspose(x: Tensor5D, filter: Tensor5D, outputShape: Array<Int>, strides: Number, pad: String): Tensor5D

    fun conv3dTranspose(x: Tensor5D, filter: Tensor5D, outputShape: Array<Int>, strides: Number, pad: Number): Tensor5D

    fun depthwiseConv2d(x: Tensor3D, filter: Tensor4D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun depthwiseConv2d(x: Tensor3D, filter: Tensor4D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun depthwiseConv2d(x: Tensor3D, filter: Tensor4D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor3D

    fun depthwiseConv2d(x: Tensor4D, filter: Tensor4D, stride: Number, pad: String, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor4D

    fun depthwiseConv2d(x: Tensor4D, filter: Tensor4D, stride: Number, pad: Number, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor4D

    fun depthwiseConv2d(x: Tensor4D, filter: Tensor4D, stride: Number, pad: dynamic, dataFormat: String = definedExternally, dilation: Number = definedExternally, dimRoundingMode: String = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: String): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: Number): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor4D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor4D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: Number): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: String): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: Number): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Array<Number>, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: String): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: Number): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Array<Number>, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: String): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: String, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: Number): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Array<Number> = definedExternally): Tensor5D

    fun maxPool3d(x: Tensor5D, filterSize: Number, strides: Number, pad: Number, dimRoundingMode: String = definedExternally, dataFormat: String = definedExternally, dilations: Number = definedExternally): Tensor5D

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: String, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Array<Number>, strides: Array<Number>, pad: Number, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: String, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Array<Number>, strides: Number, pad: Number, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: String, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Number, strides: Array<Number>, pad: Number, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Number, strides: Number, pad: String, includeBatchInIndex: Boolean = definedExternally): Json

    fun maxPoolWithArgmax(x: Tensor4D, filterSize: Number, strides: Number, pad: Number, includeBatchInIndex: Boolean = definedExternally): Json

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: String): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: Number): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: String): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: Number): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor3D

    fun pool(input: Tensor3D, windowShape: Number, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor3D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: String): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: Number): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Array<Number>, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: String): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: String, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: String, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: Number): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: Number, dilations: Array<Number> = definedExternally, strides: Number = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Array<Number> = definedExternally): Tensor4D

    fun pool(input: Tensor4D, windowShape: Number, poolingType: String, pad: Number, dilations: Number = definedExternally, strides: Number = definedExternally): Tensor4D

    fun separableConv2d(x: Tensor3D, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D, strides: Array<Number>, pad: String): Tensor3D

    fun separableConv2d(x: Tensor3D, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D, strides: Array<Number>, pad: String, dilation: Array<Number> = definedExternally, dataFormat: String = definedExternally): Tensor3D

    fun separableConv2d(x: Tensor3D, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D, strides: Array<Number>, pad: String, dilation: Number = definedExternally, dataFormat: String = definedExternally): Tensor3D

    fun separableConv2d(x: Tensor3D, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D, strides: Number, pad: String): Tensor3D

    fun separableConv2d(x: Tensor3D, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D, strides: Number, pad: String, dilation: Array<Number> = definedExternally, dataFormat: String = definedExternally): Tensor3D

    fun separableConv2d(x: Tensor3D, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D, strides: Number, pad: String, dilation: Number = definedExternally, dataFormat: String = definedExternally): Tensor3D

    //Operations / Reduction

    fun all(x: Tensor): Tensor

    fun all(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun all(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun any(x: Tensor): Tensor

    fun any(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun any(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun argMax(x: Tensor): Tensor

    fun argMax(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun argMin(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun logSumExp(x: Tensor): Tensor

    fun logSumExp(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun logSumExp(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun max(x: Tensor): Tensor

    fun max(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun max(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun mean(x: Tensor): Tensor

    fun mean(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun mean(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun min(x: Tensor): Tensor

    fun min(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun min(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun prod(x: Tensor): Tensor

    fun prod(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun prod(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun sum(x: Tensor): Tensor

    fun sum(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Tensor

    fun sum(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Tensor

    //Operations / Normalization

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor1D): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor1D): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor1D, offset: Tensor = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun batchNorm(x: Tensor, mean: Tensor1D, variance: Tensor1D, offset: Tensor1D = definedExternally, scale: Tensor1D = definedExternally, varianceEpsilon: Number = definedExternally): Tensor

    fun localResponseNormalization(x: Tensor3D, depthRadius: Number = definedExternally, bias: Number = definedExternally, alpha: Number = definedExternally, beta: Number = definedExternally): Tensor3D

    fun localResponseNormalization(x: Tensor4D, depthRadius: Number = definedExternally, bias: Number = definedExternally, alpha: Number = definedExternally, beta: Number = definedExternally): Tensor4D

    fun logSoftmax(logits: Tensor, axis: Number = definedExternally): Tensor

    fun moments(x: Tensor): Json

    fun moments(x: Tensor, axis: Number = definedExternally, keepDims: Boolean = definedExternally): Json

    fun moments(x: Tensor, axis: Array<Number> = definedExternally, keepDims: Boolean = definedExternally): Json

    fun softmax(logits: Tensor, dim: Number = definedExternally): Tensor

    fun sparseToDense(sparseIndices: Tensor, sparseValues: Tensor, outputShape: Array<Int>, defaultValue: Scalar = definedExternally): Tensor

    //Operations / Images

    val image: Image

    //Operations / RNN

    fun basicLSTMCell(forgetBias: Scalar, lstmKernel: Tensor2D, lstmBias: Tensor1D, data: Tensor2D, c: Tensor2D, h: Tensor2D): Array<Tensor2D>

    fun multiRNNCell(lstmCells: dynamic, data: Tensor2D, c: Array<*>, h: Array<*>): Array<Array<Tensor2D>>

    //Operations / Logical

    fun equal(a: Tensor, b: Tensor): Tensor?

    fun greater(a: Tensor, b: Tensor): Tensor?

    fun greaterEqual(a: Tensor, b: Tensor): Tensor?

    fun less(a: Tensor, b: Tensor): Tensor?

    fun lessEqual(a: Tensor, b: Tensor): Tensor?

    fun logicalAnd(a: Tensor, b: Tensor): Tensor

    fun logicalNot(x: Tensor): Tensor

    fun logicalOr(a: Tensor, b: Tensor): Tensor

    fun logicalXor(a: Tensor, b: Tensor): Tensor

    fun notEqual(a: Tensor, b: Tensor): Tensor?

    fun where(condition: Tensor, a: Tensor, b: Tensor): Tensor?

    fun whereAsync(condition: Tensor): Promise<Tensor2D>

    //Operations / Scan

    fun cumsum(x: Tensor, axis: Number = definedExternally, exclusive: Boolean = definedExternally, reverse: Boolean = definedExternally): Tensor

    //Operations / Evaluation

    fun confusionMatrix(labels: Tensor1D, predictions: Tensor1D, numClasses: Number): Tensor2D?

    fun inTopKAsync(predictions: Tensor, targets: Tensor, k: Number = definedExternally): Promise<Tensor?>?

    fun topk(x: Tensor, k: Number = definedExternally, sorted: Boolean = definedExternally): Json?

    //Operations / Segment

    fun unsortedSegmentSum(x: Tensor, segmentIds: Tensor1D, numSegments: Number): Tensor

    //Operations / Moving Average

    fun movingAverage(v: Tensor, x: Tensor, decay: Number): Tensor?

    fun movingAverage(v: Tensor, x: Tensor, decay: Number, step: Number = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    fun movingAverage(v: Tensor, x: Tensor, decay: Number, step: Scalar = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    fun movingAverage(v: Tensor, x: Tensor, decay: Scalar): Tensor?

    fun movingAverage(v: Tensor, x: Tensor, decay: Scalar, step: Number = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    fun movingAverage(v: Tensor, x: Tensor, decay: Scalar, step: Scalar = definedExternally, zeroDebias: Boolean = definedExternally): Tensor?

    //Operations / Slicing and Joining

    fun gatherND(x: Tensor, indices: Tensor): Tensor?

    fun scatterND(indices: Tensor, updates: Tensor, shape: Array<Int>): Tensor?

    fun stridedSlice(x: Tensor, begin: Array<Number>, end: Array<Number>, strides: Array<Number> = definedExternally, beginMask: Number = definedExternally, endMask: Number = definedExternally, ellipsisMask: Number = definedExternally, newAxisMask: Number = definedExternally, shrinkAxisMask: Number = definedExternally): Tensor?

    //Operations / Spectral

    val spectral: Spectral

    //Operations / Dropout

    fun dropout(x: Tensor, rate: Number): Tensor

    fun dropout(x: Tensor, rate: Number, noiseShape: Array<Number> = definedExternally, seed: Number = definedExternally): Tensor

    fun dropout(x: Tensor, rate: Number, noiseShape: Array<Number> = definedExternally, seed: String = definedExternally): Tensor

    //Operations / Signal

    val signal: Signal

    //Operations / Linear Algebra

    val linalg: Linalg

    //Training / Gradients

    //fun grad(f:(x: Tensor)->Tensor):(x: Tensor, dy: Tensor? = definedExternally)->Tensor

    //fun grads(f:(vararg args: Array<Tensor>)->Tensor):(args: Array<*>, dy: Tensor? = definedExternally)->Tensor

    //fun customGrad(f:dynamic):(vararg args: Array<Tensor>)->Tensor

    //fun valueAndGrad(f:(x: Tensor)->Tensor):(x: Tensor, dy: Tensor? = definedExternally)->Json

    //fun valueAndGrads(f:(vararg args: Array<Tensor>)->Tensor):(x: Array<Tensor>, dy: Tensor? = definedExternally)->Json

    fun variableGrads(f:()->Scalar, varList: Array<Variable> = definedExternally): Json

    //Training / Optimizers

    val train: Train

    //Training / Losses

    val losses: Losses

    //Performance / Memory

    fun <T> tidy(nameOrFn:()->T): T

    fun <T> tidy(nameOrFn: String, fn:()->T): T

    fun dispose(container: Unit)

    fun dispose(container: Number)

    fun dispose(container: String)

    fun dispose(container: Tensor)

    fun dispose(container: Array<Tensor>)

    fun dispose(container: Json)

    fun dispose(container: dynamic)

    fun keep(result: Tensor)

    fun memory(): MemoryInfo

    //Performance / Timing

    fun time(f:()->Unit): Promise<TimingInfo>

    fun nextFrame(): Promise<Unit>

    //Performance / Profile

    fun profile(f:()->Unit): Promise<ProfileInfo>

    fun profile(f:()->Number): Promise<ProfileInfo>

    fun profile(f:()->String): Promise<ProfileInfo>

    fun profile(f:()->Tensor): Promise<ProfileInfo>

    fun profile(f:()->Array<Tensor>): Promise<ProfileInfo>

    fun profile(f:()->Json): Promise<ProfileInfo>

    //Environment

    fun disposeVariables()

    fun enableDebugMode()

    fun enableProdMode()

    fun engine(): Engine

    fun env(): dynamic // https://js.tensorflow.org/api/latest/#env

    //Constraints

    val constraints: Constraints

    //Initializers

    val initializers: Initializers

    //Regularizers

    val regularizers: Regularizers

    //Data

    val data: Data

    //Util

    val util: Util

    //Browser

    val browser: Browser

    //Backends

    fun backend(): KernelBackend

    fun getBackend(): String

    fun ready(): Promise<Unit>

    fun registerBackend(name: String, factory:()->KernelBackend, priority: Number = definedExternally): Boolean

    fun removeBackend(name: String)

    fun setBackend(backendName: String): Promise<Boolean>

    //Metrics

    val metrics: Metrics

    //Callbacks

    val callbacks: Callbacks

    //IO

    val io: IO

}

val tf = Tensorflow

//val tfvis = require("@tensorflow/tfjs-vis")
/**
 * @tensorflow/tfjs-vis CDN
 * 因使用以NPM安裝嘅tfjs-vis出問題
 * 解決方法暫時用CDN版本住先
 * */
val tfvis = js("tfvis")