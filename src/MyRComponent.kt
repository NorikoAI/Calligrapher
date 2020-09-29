package org.sourcekey.NorikoAI.Calligrapher

import org.w3c.dom.Element
import org.w3c.dom.HTMLDivElement
import react.*
import react.dom.div
import react.dom.findDOMNode


abstract class MyRComponent<P: RProps, S: RState>(props: P): RComponent<P, S>(props) {

    var thisRef: dynamic = null

    val thisElement: Element
        get() = findDOMNode(thisRef)

    open fun onVisible(){}

    open fun onInvisible(){}

    abstract fun RBuilder.myRender()

    override fun RBuilder.render() {
        div {
            ref { thisRef = it }
            myRender()
        }
    }

    init {
        jq(thisElement).bind("isOnVisible", fun(event){
            onVisible()
        })
        jq(thisElement).show("slow", fun(){
            jqThis().trigger("isOnVisible")
        })
    }
}