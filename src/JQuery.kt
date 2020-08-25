package org.sourcekey.NorikoAI.Calligrapher
/*
import jquery.JQuery
import org.w3c.dom.Element
import org.w3c.dom.events.Event
import org.w3c.xhr.XMLHttpRequest
import kotlin.js.Json


/**
 * 導入JQuery
 *
 * 響JavaSricpt度導入JQuery去"$"
 * 使Kotlin編碼能夠Call "$" 來使用JQuery
 * */
private val importJQuery = js("var \$ = require(\"jquery\")")

/**
 * 設立純JQuery值 "$"
 *
 * 為有時侯需要 $.fff(v); 咁寫而設
 * */
inline val jq: JQuery
    get() = js("\$")

/**
 * 等於 $(this) JS寫法
 *
 * 必要寫inline fun原因係為左編譯為JS時寫住 $(this)
 * */
inline fun jqThis(): JQuery = js("\$")(js("this"))

inline val JQuery.length: Int
    get() = asDynamic().length

inline fun JQuery.on(events: String, noinline handler: (Event)->Unit): JQuery = asDynamic().on(events, handler)

inline fun JQuery.on(events: String, selector: String, noinline handler: (Event)->Unit): JQuery = asDynamic().on(events, selector, handler)

inline fun JQuery.on(events: String, selector: String, data: dynamic, noinline handler: (Event)->Unit): JQuery = asDynamic().on(events, selector, data, handler)

inline fun JQuery.on(events: String): JQuery = asDynamic().on(events)

inline fun JQuery.on(events: String, selector: String): JQuery = asDynamic().on(events, selector)

inline fun JQuery.on(events: String, selector: String, data: dynamic): JQuery = asDynamic().on(events, selector, data)

inline fun JQuery.index(): Int = asDynamic().index()

inline fun JQuery.index(selector: String): Int = asDynamic().index(selector)

inline fun JQuery.index(element: Element): Int = asDynamic().index(element)

inline fun JQuery.index(element: JQuery): Int = asDynamic().index(element)

inline fun JQuery.eq(index: Int): JQuery = asDynamic().eq(index)

inline fun JQuery.focus(): JQuery = asDynamic().focus()

inline fun JQuery.focus(noinline handler: (Event)->Unit): JQuery = asDynamic().focus(handler)

inline fun JQuery.focus(eventData: dynamic, noinline handler: (Event)->Unit): JQuery = asDynamic().focus(eventData, handler)

inline fun JQuery.scrollTop(): Double = asDynamic().scrollTop()

inline fun JQuery.scrollTop(value: Double): JQuery = asDynamic().scrollTop(value)

inline fun JQuery.scrollLeft(): Double = asDynamic().scrollLeft()

inline fun JQuery.scrollLeft(value: Double): JQuery = asDynamic().scrollLeft(value)

inline fun JQuery.mouseleave(): JQuery = asDynamic().mouseleave()

inline fun JQuery.mouseleave(noinline handler: (Event)->Unit): JQuery = asDynamic().mouseleave(handler)

inline fun JQuery.mouseleave(eventData: dynamic, noinline handler: (Event)->Unit): JQuery = asDynamic().mouseleave(eventData, handler)

inline fun JQuery.mouseout(): JQuery = asDynamic().mouseout()

inline fun JQuery.mouseout(noinline handler: (Event)->Unit): JQuery = asDynamic().mouseout(handler)

inline fun JQuery.mouseout(eventData: dynamic, noinline handler: (Event)->Unit): JQuery = asDynamic().mouseout(eventData, handler)

inline fun JQuery.tabPrev(): JQuery = asDynamic().tabPrev()

inline fun JQuery.tabNext(): JQuery = asDynamic().tabNext()

inline fun JQuery.css(propertyName: String): String = asDynamic().css(propertyName)

inline fun JQuery.css(propertyNames: Array<String>): String = asDynamic().css(propertyNames)

inline fun JQuery.css(propertyName: String, value: Double): JQuery = asDynamic().css(propertyName, value)

inline fun JQuery.css(propertyName: String, value: String): JQuery = asDynamic().css(propertyName, value)

inline fun JQuery.css(propertyName: String, noinline function: (index: Int, value: String)->String): JQuery = asDynamic().css(propertyName, function)

inline fun JQuery.css(properties: Json): JQuery = asDynamic().css(properties)

inline fun JQuery.prev(selector: String): JQuery = asDynamic().prev(selector)

inline fun JQuery.children(selector: String): JQuery = asDynamic().children(selector)

inline fun JQuery.find(selector: String): JQuery = asDynamic().find(selector)

inline fun JQuery.find(selector: Element): JQuery = asDynamic().find(selector)

inline fun JQuery.find(selector: JQuery): JQuery = asDynamic().find(selector)

inline fun JQuery.get(): Element? = asDynamic().get()

inline fun JQuery.get(index: Int): Element? = asDynamic().get(index)

inline fun JQuery.ajax(url: String): XMLHttpRequest = asDynamic().ajax(url)

inline fun JQuery.ajax(url: String, settings: Json): XMLHttpRequest = asDynamic().ajax(url, settings)

inline fun JQuery.ajax(settings: Json): XMLHttpRequest = asDynamic().ajax(settings)
*/
/*
inline fun JQuery.iS(selector: String): Boolean = {
    val _this = asDynamic()
    val _selector = selector
    js("_this.is(_selector)") as Boolean
}()

inline fun JQuery.iS(noinline function: (index: Int, element: Element)->Boolean): Boolean = {
    val _this = asDynamic()
    val _function = function
    js("_this.is(_function)") as Boolean
}()

inline fun JQuery.iS(selection: JQuery): Boolean = {
    val _this = asDynamic()
    val _selection = selection
    js("_this.is(_selection)") as Boolean
}()

inline fun JQuery.iS(element: Element): Boolean = {
    val _this = asDynamic()
    val _element = element
    js("_this.is(_element)") as Boolean
}()
*/