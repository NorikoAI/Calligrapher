package org.sourcekey.NorikoAI.Calligrapher

import app.wcl01Url
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
fun String.codePointAt(index: Int): Int? = asDynamic().codePointAt(index)

/**
 *
 * */
fun String.Companion.fromCharPoint(unicode: Int): String = asDynamic().fromCharPoint(unicode)

/**
 *
 * */
fun Object.values(obj: dynamic): Array<dynamic> = asDynamic().values(obj)

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

