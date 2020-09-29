package index

import app.*
import kotlinext.js.*
import react.dom.*
import kotlinx.browser.*
import react.functionalComponent
import react.infinite.grid.InfiniteGrid

fun example(index: Int, key: String = "") = functionalComponent<dynamic> {
    div {
        +"This is ${index}"
    }
}

fun main(){
    requireAll(require.context("src", true, js("/\\.css$/")))
    render(document.getElementById("root")) {
        app()
        /*
        InfiniteGrid{
            attrs.wrapperHeight = 400
            attrs.entries = arrayOf(example(1), example(2), example(3), example(4))
        }*/
    }
}