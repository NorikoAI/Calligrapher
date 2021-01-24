package org.sourcekey.NorikoAI.Calligrapher

import ExtendedFun.drawNumbersOfRange
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.css.*
import kotlinx.html.InputType
import kotlinx.html.js.onChangeFunction
import kotlinx.html.js.onClickFunction
import logo.logo
import org.w3c.dom.events.Event
import react.RBuilder
import react.RComponent
import react.RProps
import react.RState
import react.dom.button
import react.dom.code
import react.dom.defaultValue
import react.dom.h2
import styled.css
import styled.styledDiv
import styled.styledInput
import styled.styledP
import kotlin.random.Random


class Menu : RComponent<RProps, RState>() {
    override fun RBuilder.render() {
        styledDiv {
            css {
                backgroundColor = Color("#000")
                height = 100.px
                padding = "20px"
                color = Color.white
            }
            logo()
            h2 {
                +"Welcome to React with Kotlin"
            }
        }
        styledP {
            css {
                fontSize = LinearDimension("large")
            }
            +"To get started, edit "
            code { +"App.kt" }
            +" and save to reload."
        }
        styledP {
            css {
                fontSize = LinearDimension("medium")
            }
            ticker()
        }
        //project(getProject)





        styledInput(type = InputType.number) {
            attrs {
                val f  = fun(event: Event){
                    project.calligrapher.sectionSize =
                        (event.target.asDynamic().value as? String)?.toIntOrNull()?:return
                }
                onChangeFunction = f
                defaultValue = project.calligrapher.sectionSize.toString()
            }
        }
        styledInput(type = InputType.number) {
            attrs {
                val f  = fun(event: Event){
                    project.calligrapher.epochs =
                        (event.target.asDynamic().value as? String)?.toIntOrNull()?:return
                }
                onChangeFunction = f
                defaultValue = project.calligrapher.epochs.toString()
            }
        }
        button {
            attrs.onClickFunction = fun(event){
                for(f in project.font){ console.log(f) }
            }
            +"FFF"
        }
        /*button {
            attrs.onClickFunction = fun(event){
                //project.initModel()
                println("Init Model Done")
            }
            +"Init Model"
        }*/
        button {
            attrs.onClickFunction = fun(event){
                GlobalScope.launch {
                    project.calligrapher.train()
                    println("Train Model Done")
                }
            }
            +"Train Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                for(f in project.font){ f?.isKeep = false }
            }
            +"C S"
        }
        button {
            attrs.onClickFunction = fun(event){
                for(i in 35..57){ project.font.getGlyphByUnicode(i)?.isKeep = true }
                project.calligrapher.train()
            }
            +"Q T"
        }
        button {
            attrs.onClickFunction = fun(event){
                for(f in project.font){ f?.isKeep = false }
                val indexs = Random.drawNumbersOfRange(35, 70, 10)
                for(i in indexs){
                    println(i)
                    console.log(project.font[i])
                    console.log(project.font[i]?.path)
                    project.font[i]?.isKeep = true
                }
                project.calligrapher.train()
            }
            +"R T"
        }
        var unicode = 57
        styledInput(type = InputType.number) {
            attrs {
                val f  = fun(event: Event){
                    unicode = (event.target.asDynamic().value as? String)?.toIntOrNull()?:return
                }
                onChangeFunction = f
                defaultValue = unicode.toString()
            }
        }
        button {
            attrs.onClickFunction = fun(event){
                GlobalScope.launch {
                    println("S")
                    project.calligrapher.predict(unicode)
                    println("F")
                }
            }
            +"Predict"
        }
        button {
            attrs.onClickFunction = fun(event){
                GlobalScope.launch {
                    project.calligrapher.write(unicode)
                    println("Write Done")
                }
            }
            +"Write"
        }
        button {
            attrs.onClickFunction = fun(event){
                GlobalScope.launch {
                    project.calligrapher.modelManager.save()
                    println("Save Model Done")
                }
            }
            +"Save Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                GlobalScope.launch {
                    project.calligrapher.modelManager
                    println("Load Model Done")
                }
            }
            +"Load Model"
        }
        button {
            attrs.onClickFunction = fun(event){
                console.log("Download Model")
                //model?.save("downloads://my-model")
            }
            +"Download Model"
        }
    }
}

fun RBuilder.menu() = child(Menu::class) {}