package ExtendedFun

import kotlinext.js.Object
import kotlin.js.Json
import kotlin.random.Random


/**
 *
 * */
inline fun jsObject(init: dynamic.() -> Unit): Json {
    val obj = js("{}")
    init(obj)
    return obj
}

/**
 *
 * */
fun <T> Object.Companion.values(obj: dynamic): Array<T> = asDynamic().values(obj)

/**
 *
 * */
fun Object.Companion.size(obj: dynamic): Int = getOwnPropertyNames(obj).size

/**
 *
 * */
fun Object.Companion.delete(obj: dynamic, property: String){
    val _obj = obj
    val _prop = property
    js("delete _obj[_prop]")
}

/**
 *
 * */
operator fun <T> Object.get(property: String): T? = asDynamic()[property] as? T

/**
 *
 * */
operator fun <T> Object.get(index: Int): T? = get<T>(Object.getOwnPropertyNames(this)[index])

/**
 *
 * */
operator fun <T> Object.iterator(): Iterator<T> = object : Iterator<T>{
    val array = Object.values<T>(this@iterator)
    val size = Object.size(this@iterator)
    var i = 0
    override fun hasNext(): Boolean = i < size
    override fun next(): T = array[i++]
}

/**
 *
 * */
fun <Value> Value.equals(values: Array<Value>): Boolean{
    values.forEach { value -> if(value == this){
        return@equals true
    }}
    return false
}

/**
 *
 * */
fun <Value> Value.equals(vararg values: Value): Boolean{
    values.forEach { value -> if(value == this){
        return@equals true
    }}
    return false
}

/**
 *
 * UTF-32
 * */
fun String.codePointAt(index: Int): Int? = asDynamic().codePointAt(index)

/**
 *
 * UTF-16
 * */
fun String.charCodeAt(index: Int): Int? = asDynamic().charCodeAt(index)

/**
 *
 * UTF-32
 * */
fun String.Companion.fromCodePoint(unicode: Int): String = js("String").fromCodePoint(unicode)

/**
 *
 * UTF-16
 * */
fun String.Companion.fromCharCode(unicode: Int): String = js("String").fromCharCode(unicode)

fun Random.range(min: Int, max: Int): Int?{
    return try { nextInt(min, max) }catch (e:dynamic){ null }
}

/**
 *
 * */
fun Random.drawNumbersOfRange(min: Int, max: Int, numberOfValue: Int): Array<Int>{
    val drawedNumbers = ArrayList<Int>()
    var _numberOfValue = numberOfValue
    if(max < _numberOfValue){_numberOfValue = max}
    val numberList = ArrayList<Int>()
    for(i in 0 until max){numberList.add(i)}
    for(i in 0 until numberOfValue){
        val drawedIndex = this.nextInt(min, max-1)
        drawedNumbers.add(numberList[drawedIndex])
        numberList.remove(drawedIndex)
    }
    drawedNumbers.sort()
    return drawedNumbers.toTypedArray()
}

/**
 *
 * */
fun <T> Array<out T>.findFromIndexStart(startIndex: Int, predicate: (index: Int, element: T)->Boolean): T?{
    var i = startIndex
    while (i < this.size){
        val element = this.getOrNull(i)?:return null
        if (predicate(i, element)) return element
        i++
    }
    i = 0
    while (i < startIndex){
        val element = this.getOrNull(i)?:return null
        if (predicate(i, element)) return element
        i++
    }
    return null
}

/**
 *
 * */
fun <T> List<T>.findFromIndexStart(startIndex: Int, predicate: (index: Int, element: T)->Boolean): T?{
    var i = startIndex
    while (i < this.size){
        val element = this.getOrNull(i)?:return null
        if (predicate(i, element)) return element
        i++
    }
    i = 0
    while (i < startIndex){
        val element = this.getOrNull(i)?:return null
        if (predicate(i, element)) return element
        i++
    }
    return null
}