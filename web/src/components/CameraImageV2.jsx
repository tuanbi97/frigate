import { h } from 'preact';
import ActivityIndicator from './ActivityIndicator';
import { useApiHost } from '../api';
import useSWR from 'swr';
import { useCallback, useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { useResizeObserver } from '../hooks';

export default function CameraImageV2({ camera, onload, searchParams = '', stretch = false, frameInterval = 200 }) {
  const { data: config } = useSWR('config');
  const apiHost = useApiHost();
  const [hasLoaded, setHasLoaded] = useState(false);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [{ width: containerWidth }] = useResizeObserver(containerRef);
  const [preloadedImages, setPreloadedImages] = useState([]);
  const [waitForPreload, setWaitForPreload] = useState(0);
  const [key, setKey] = useState(Date.now());
  const waitForMinFrames = Math.round(3000 / frameInterval);
  const maxBufferFrames = Math.round(6000 / frameInterval);

  // Add scrollbar width (when visible) to the available observer width to eliminate screen juddering.
  // https://github.com/blakeblackshear/frigate/issues/1657
  let scrollBarWidth = 0;
  if (window.innerWidth && document.body.offsetWidth) {
    scrollBarWidth = window.innerWidth - document.body.offsetWidth;
  }
  const availableWidth = scrollBarWidth ? containerWidth + scrollBarWidth : containerWidth;

  const { name } = config ? config.cameras[camera] : '';
  const enabled = config ? config.cameras[camera].enabled : 'True';
  const { width, height } = config ? config.cameras[camera].detect : { width: 1, height: 1 };
  const aspectRatio = width / height;

  const scaledHeight = useMemo(() => {
    const scaledHeight = Math.floor(availableWidth / aspectRatio);
    const finalHeight = stretch ? scaledHeight : Math.min(scaledHeight, height);

    if (finalHeight > 0) {
      return finalHeight;
    }

    return 100;
  }, [availableWidth, aspectRatio, height, stretch]);
  const scaledWidth = useMemo(
    () => Math.ceil(scaledHeight * aspectRatio - scrollBarWidth),
    [scaledHeight, aspectRatio, scrollBarWidth]
  );

  useEffect(() => {
    const loadTime = Date.now() - key;
    console.log("Start use Effect", Date.now());
    const intervalId = setTimeout(() => {
      setKey(Date.now());
      console.log("Start timeOut", Date.now());
      console.log(preloadedImages.length);
      if (waitForPreload) {
        setWaitForPreload(Math.max(0, waitForMinFrames - preloadedImages.length));
        return;
      }
      if (canvasRef.current) {
        if (preloadedImages.length) {
          const ctx = canvasRef.current.getContext('2d');
          ctx.drawImage(preloadedImages[0], 0, 0, scaledWidth, scaledHeight);
          // Free up resources by removing the displayed frame
          setPreloadedImages((prevImages) => {
            const newImages = [...prevImages];
            newImages.shift(); // Remove the displayed frame
            return newImages;
          });
        }
        else {
          setWaitForPreload(waitForMinFrames);
          return;
        }
      }
    }, preloadedImages.length > maxBufferFrames ? 1 : frameInterval);

    return () => clearTimeout(intervalId);
  }, [key]);

  useEffect(() => {
    if (!config || scaledHeight === 0) {
      return;
    }
    console.log(searchParams);
    fetch(`${apiHost}api/${name}/latest.jpg?h=${scaledHeight}${searchParams ? `&${searchParams}` : ''}`)
      .then(res => {
        return res.status == 204 ? Promise.resolve(null) : res.blob()
      })
      .then(blob => {
        if (blob) {
          const img = new Image();
          img.onload = (event) => {
            setHasLoaded(true);
            setPreloadedImages((prevImages) => [...prevImages, img]);
            onload && onload("loaded");
          };
          img.src = URL.createObjectURL(blob);
        }
        else {
          onload && onload("waiting");
        }
      })
      .catch(error => {
        console.log(error);
      });
  }, [apiHost, name, searchParams, scaledHeight, config, onload]);

  return (
    <div className="relative w-full" ref={containerRef}>
      {enabled ? (
        <canvas data-testid="cameraimage-canvas" height={scaledHeight} ref={canvasRef} width={scaledWidth} />
      ) : (
        <div class="text-center pt-6">Camera is disabled in config, no stream or snapshot available!</div>
      )}
      {(!hasLoaded || waitForPreload) && enabled ? (
        <div className="absolute inset-0 flex justify-center" style={`height: ${scaledHeight}px`}>
          <ActivityIndicator />
        </div>
      ) : null}
    </div>
  );
}
