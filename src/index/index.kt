package index

import app.*
import kotlinext.js.*
import react.dom.*
import kotlinx.browser.*


fun main(){
    requireAll(require.context("src", true, js("/\\.css$/")))
    render(document.getElementById("root")) {
        app()
    }
}



