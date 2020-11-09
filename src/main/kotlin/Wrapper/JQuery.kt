package JQuery

import kotlinext.js.Object
import org.w3c.dom.Element
import org.w3c.dom.events.Event
import org.w3c.xhr.XMLHttpRequest
import kotlin.js.Json


open external class MouseEvent {
    val pageX: Double
    val pageY: Double
    fun preventDefault()
    fun isDefaultPrevented(): Boolean
}

external class MouseClickEvent : MouseEvent {
    val which: Int
}

external class JQuery{
    fun addClass(className: String): JQuery
    fun addClass(f: (Int, String) -> String): JQuery

    fun attr(attrName: String): String
    fun attr(attrName: String, value: String): JQuery

    fun html(): String
    fun html(s: String): JQuery
    fun html(f: (Int, String) -> String): JQuery


    fun hasClass(className: String): Boolean
    fun removeClass(className: String): JQuery
    fun height(): Number
    fun width(): Number

    fun click(): JQuery

    fun mousedown(handler: (MouseEvent) -> Unit): JQuery
    fun mouseup(handler: (MouseEvent) -> Unit): JQuery
    fun mousemove(handler: (MouseEvent) -> Unit): JQuery

    fun dblclick(handler: (MouseClickEvent) -> Unit): JQuery
    fun click(handler: (MouseClickEvent) -> Unit): JQuery

    fun load(handler: () -> Unit): JQuery
    fun change(handler: () -> Unit): JQuery

    fun append(str: String): JQuery
    fun ready(handler: () -> Unit): JQuery
    fun text(text: String): JQuery
    fun slideUp(): JQuery
    fun hover(handlerInOut: () -> Unit): JQuery
    fun hover(handlerIn: () -> Unit, handlerOut: () -> Unit): JQuery
    fun next(): JQuery
    fun parent(): JQuery
    fun `val`(): String?

    ////////////////////////////////////////////////////////////////
    fun `is`(selector: String): Boolean
    fun `is`(function: (index: Int, element: Element)->Boolean): Boolean
    fun `is`(selection: JQuery): Boolean
    fun `is`(element: Element): Boolean

    fun on(events: String, handler: (Event)->Unit): JQuery
    fun on(events: String, selector: String, handler: (Event)->Unit): JQuery
    fun on(events: String, selector: String, data: dynamic, handler: (Event)->Unit): JQuery
    fun on(events: String): JQuery
    fun on(events: String, selector: String): JQuery
    fun on(events: String, selector: String, data: dynamic): JQuery

    fun index(): Int
    fun index(selector: String): Int
    fun index(element: Element): Int
    fun index(element: JQuery): Int

    fun eq(index: Int): JQuery

    fun focus(): JQuery
    fun focus(handler: (Event)->Unit): JQuery
    fun focus(eventData: dynamic, handler: (Event)->Unit): JQuery

    fun scrollTop(): Double
    fun scrollTop(value: Double): JQuery
    fun scrollLeft(): Double
    fun scrollLeft(value: Double): JQuery

    fun mouseleave(): JQuery
    fun mouseleave(handler: (Event)->Unit): JQuery
    fun mouseleave(eventData: dynamic, handler: (Event)->Unit): JQuery
    fun mouseout(): JQuery
    fun mouseout(handler: (Event)->Unit): JQuery
    fun mouseout(eventData: dynamic, handler: (Event)->Unit): JQuery

    fun tabPrev(): JQuery
    fun tabNext(): JQuery

    fun css(propertyName: String): String
    fun css(propertyNames: Array<String>): String
    fun css(propertyName: String, value: Double): JQuery
    fun css(propertyName: String, value: String): JQuery
    fun css(propertyName: String, function: (index: Int, value: String)->String): JQuery
    fun css(properties: Json): JQuery

    fun prev(selector: String): JQuery
    fun children(selector: String): JQuery
    fun find(selector: String): JQuery
    fun find(selector: Element): JQuery
    fun find(selector: JQuery): JQuery
    fun get(): Element?
    fun get(index: Int): Element?

    fun bind(eventType: String, handler: (Event) -> Unit): JQuery
    fun bind(eventType: String, eventData: dynamic = definedExternally, handler: (Event) -> Unit): JQuery
    fun bind(eventType: String, eventData: dynamic = definedExternally, preventBubble: Boolean = definedExternally): JQuery
    fun bind(events: Object): JQuery

    fun show()
    fun show(duration: Number = definedExternally, complete: ()->Unit = definedExternally): JQuery
    fun show(duration: String = definedExternally, complete: ()->Unit = definedExternally): JQuery
    fun show(options: Object): JQuery
    fun show(duration: Number = definedExternally, easing: String, complete: ()->Unit = definedExternally): JQuery
    fun show(duration: String = definedExternally, easing: String, complete: ()->Unit = definedExternally): JQuery

    fun trigger(eventType: String): JQuery
    fun trigger(eventType: String, extraParameters: Array<Object> = definedExternally): JQuery
    fun trigger(eventType: String, extraParameters: Object = definedExternally): JQuery
    fun trigger(event: Event): JQuery
    fun trigger(event: Event, extraParameters: Array<Object> = definedExternally): JQuery
    fun trigger(event: Event, extraParameters: Object = definedExternally): JQuery

    fun ajax(url: String): XMLHttpRequest
    fun ajax(url: String, settings: Json): XMLHttpRequest
    fun ajax(settings: Json): XMLHttpRequest

    val length: Int
}


/**
 * 導入JQuery
 *
 * 響JavaSricpt度導入JQuery去"$"
 * 使Kotlin編碼能夠Call "$" 來使用JQuery
 * */
private val importJQuery = js("var \$ = require(\"jquery\")")

@JsName("$")
external fun jq(selector: String): JQuery
@JsName("$")
external fun jq(selector: String, context: Element): JQuery
@JsName("$")
external fun jq(callback: () -> Unit): JQuery
@JsName("$")
external fun jq(obj: JQuery): JQuery
@JsName("$")
external fun jq(el: Element): JQuery
@JsName("$")
external fun jq(): JQuery
@JsName("$")
external val jq: JQuery //設立純JQuery值 "$" 為有時侯需要 $.foo(v); 咁寫而設
/**
 * 等於 $(this) JS寫法
 *
 * 必要寫external fun原因係為左編譯為JS時寫住 $(this)
 * */
inline fun jqThis(): JQuery = js("\$")(js("this"))