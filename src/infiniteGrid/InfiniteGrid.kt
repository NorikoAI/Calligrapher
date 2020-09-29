@file:JsModule("react-infinite-grid")
@file:JsNonModule
package react.infinite.grid

import react.*



external interface InfiniteGridProps: RProps {
    var wrapperHeight: Int
    var entries: Array<FunctionalComponent<dynamic>>
}

@JsName("default")
external val InfiniteGrid : RClass<InfiniteGridProps>




























/*
package react.infinite.grid

import react.*
import react.dom.div

external interface InfiniteGridItemProps: RProps {
    var index: Int
    var key: String
}

external interface InfiniteGridProps: RProps {
    var wrapperHeight: Int
    var entries: Array<FunctionalComponent<InfiniteGridItemProps>>
}

//private external val reactInfiniteGrid: dynamic

@JsModule("react-infinite-grid")
@JsNonModule
@JsName("InfiniteGrid")
external val InfiniteGrid : RClass<InfiniteGridProps>







fun myComplexComponent(index: Int, key: String = "") = functionalComponent<InfiniteGridItemProps> {
    println("2")
    +"${index}"
}

val aaa = arrayOf(myComplexComponent(1,"1"))


fun RBuilder.infiniteGrid(wrapperHeight: Int = 400, entities: Array<FunctionalComponent<InfiniteGridItemProps>> = aaa){
    InfiniteGrid{
        val aaa = arrayOf(myComplexComponent(1,"1"))
        attrs.wrapperHeight = 400
        attrs.entries = aaa
    }
}*/