plugins {
    id("org.jetbrains.kotlin.js") version "1.4.21"
}

group = "package org.sourcekey.NorikoAI.Calligrapher"
version = "1.0.0"

repositories {
    maven("https://kotlin.bintray.com/kotlin-js-wrappers/")
    mavenCentral()
    jcenter()
}

dependencies {
    implementation(kotlin("stdlib-js"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.3.9")
    implementation("org.jetbrains:kotlin-extensions:1.0.1-pre.110-kotlin-1.4.0")
    //React, React DOM + Wrappers (chapter 3)
    implementation("org.jetbrains:kotlin-react:16.13.1-pre.110-kotlin-1.4.0")
    implementation("org.jetbrains:kotlin-react-dom:16.13.1-pre.110-kotlin-1.4.0")
    implementation(npm("react", "16.13.1"))
    implementation(npm("react-dom", "16.13.1"))
    //implementation(npm("react-is", "16.13.1"))
    //Video Player Share Buttons
    implementation(npm("react-player", "~2.6.0"))
    implementation(npm("react-share", "~4.2.1"))
    //Kotlin Styled (chapter 3)
    implementation("org.jetbrains:kotlin-css:1.0.0-pre.110-kotlin-1.4.0")
    implementation("org.jetbrains:kotlin-styled:1.0.0-pre.110-kotlin-1.4.0")
    implementation(npm("styled-components", "4.4.0"))
    implementation(npm("inline-style-prefixer", "6.0.0"))
    //MaterialUI Muirwik
    implementation(npm("@material-ui/core", "^4.11.0"))
    implementation(npm("@material-ui/icons", "^4.9.1"))
    implementation("com.ccfraser.muirwik:muirwik-components:0.6.0")
    //egjs/react-infinitegrid
    implementation(npm("@egjs/react-infinitegrid", "~3.0.5"))
    //JQuery
    val jQueryVersion = "3.5.1"
    implementation(npm("jquery", jQueryVersion))
    //implementation(npm("@types/jquery", jQueryVersion, generateExternals = true))
    //Opentype.js
    val opentypeJsVersion = "1.3.3"
    implementation(npm("opentype.js", opentypeJsVersion))
    //implementation(npm("@types/opentype.js", "1.3.1", generateExternals = true))
    //TensorFlow
    implementation(npm("@tensorflow/tfjs", "3.2.0", generateExternals = false))
    implementation(npm("@tensorflow/tfjs-vis", "1.4.3", generateExternals = false))
    //D3
    val d3Version = "6.2.0"
    implementation(npm("d3", d3Version))
    //implementation(npm("@types/d3", d3Version, generateExternals = true))
}

kotlin {
    js/*(IR)*/ {
        browser {
            webpackTask {
                cssSupport.enabled = true
            }

            runTask {
                cssSupport.enabled = true
            }

            testTask {
                useKarma {
                    useChromeHeadless()
                    webpackConfig.cssSupport.enabled = true
                }
            }
        }
        binaries.executable()

        useCommonJs()
    }
}