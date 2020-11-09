package org.sourcekey.NorikoAI.Calligrapher

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

fun <Value> Value.equals(values: Array<Value>): Boolean{
    values.forEach { value -> if(value == this){
        return@equals true
    }}
    return false
}

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

/**
 *
 * */
@JsName("Object")
external object JsObject

/**
 *
 * */
fun JsObject.values(obj: dynamic): Array<dynamic> = asDynamic().values(obj)

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

