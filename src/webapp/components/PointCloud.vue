<script lang="ts">
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
	acceleratedRaycast, computeBoundsTree, disposeBoundsTree, MeshBVHVisualizer, INTERSECTED, NOT_INTERSECTED,
	SAH, CENTER, AVERAGE,
} from 'three-mesh-bvh';
import { render } from 'nuxt/dist/app/compat/capi';

THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export default defineComponent({
    
    name: 'PointCloud',
    setup() {

        const canvas = ref(null);
    
        

        return {
            canvas

        };
    },
    mounted() {
        console.log('Component mounted.');
        // const pointCloud = new THREE.Group();
        const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 2000 );
        const scene = new THREE.Scene();
        const renderer = new THREE.WebGLRenderer();
        let control = null;
        const init = (() => {;
            scene.background = new THREE.Color( 'D9DBDE' );
            camera.position.set( 20, 0, 0 );
            camera.lookAt( scene.position );
            renderer.setSize( window.innerWidth, window.innerHeight );
            renderer.setPixelRatio( window.devicePixelRatio );
            renderer.autoClear = false;
            renderer.shadowMap.enabled = true;
            const pointCloudCanvas = this.$refs['point-cloud-canvas'];
            if (pointCloudCanvas instanceof HTMLElement) {
                pointCloudCanvas.appendChild( renderer.domElement );
                control = new OrbitControls( camera, renderer.domElement );
            }
            
        });
        
        init(); 

        const torchSLAM = useTorchSLAM();

        const xyz = () => {
            const xyz = torchSLAM.value.landmark_locations.xyz;
            return xyz;
        };
        
        const buildPointCloudGeometry = () => {


            const pointCloud = xyz();
            const geometry = new THREE.BufferGeometry();
            const numPoints = pointCloud.length;

            const positions = (() => {
                const xyz = new Float32Array( numPoints * 3 );
                xyz.set(pointCloud.flat());
                return xyz;
            })
            const colors = (() => {
                const iterable = (function* () {
                  yield* [37, 38, 41];
                })();
                const rgb = new Float32Array( iterable );
                return rgb;
            })

            geometry.setAttribute( 'position', new THREE.BufferAttribute( positions(), 3 ) );
            geometry.setAttribute( 'color', new THREE.BufferAttribute( colors(), 3 ) );
            geometry.computeBoundingSphere();

            return geometry;

        }

        const pointSize = 0.15;


        const buildPointCloud = () => {
            const geometry = buildPointCloudGeometry();
            const material = new THREE.PointsMaterial( { size: pointSize, vertexColors: true } );
            const pcd = new THREE.Points( geometry, material );
            return pcd;
        }
        
        const renderPointCloud = () => {
            const pcd = buildPointCloud();
            scene.add( pcd );
        };
        
        const animate = () => {
            requestAnimationFrame( animate );
            renderPointCloud();
            renderer.render( scene, camera );

        };

        animate();

    }
})
</script>
<template>
    <div ref="point-cloud-canvas">
    </div>
</template>